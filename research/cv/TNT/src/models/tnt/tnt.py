# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Transformer in Transformer(TNT)"""
import math

import numpy as np
import mindspore.common.initializer as weight_init
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype

from .misc import DropPath1D, to_2tuple, Identity, trunc_array


def make_divisible(v, divisor=8, min_value=None):
    """make_divisible"""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class UnfoldKernelEqPatch(nn.Cell):
    """UnfoldKernelEqPatch with better performance"""

    def __init__(self, kernel_size, strides):
        super(UnfoldKernelEqPatch, self).__init__()
        assert kernel_size == strides
        self.kernel_size = kernel_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, inputs):
        B, C, H, W = inputs.shape
        inputs = self.reshape(inputs,
                              (B, C, H // self.kernel_size[0], self.kernel_size[0], W))
        inputs = self.transpose(inputs, (0, 2, 1, 3, 4))
        inputs = self.reshape(inputs, (-1, C, self.kernel_size[0], W // self.kernel_size[1], self.kernel_size[1]))
        inputs = self.transpose(inputs, (0, 3, 1, 2, 4))
        inputs = self.reshape(inputs, (-1, C, self.kernel_size[0], self.kernel_size[1]))

        return inputs


class Mlp(nn.Cell):
    """Mlp"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=False)
        self.drop = nn.Dropout(keep_prob=1.0 - drop) if drop > 0. else Identity()

    def construct(self, x):
        x = self.fc1(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE(nn.Cell):
    """SE Block"""

    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.SequentialCell([
            LayerNorm(normalized_shape=dim, eps=1e-05),
            nn.Dense(in_channels=dim, out_channels=hidden_dim, has_bias=False),
            nn.ReLU(),
            nn.Dense(in_channels=hidden_dim, out_channels=dim, has_bias=False),
            nn.Tanh()
        ])

    def construct(self, x):
        a = P.ReduceMean()(True, x, 1)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Attention(nn.Cell):
    """Attention"""

    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qk = nn.Dense(in_channels=dim, out_channels=hidden_dim * 2, has_bias=qkv_bias)
        self.q = nn.Dense(in_channels=dim, out_channels=hidden_dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=hidden_dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=False)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = P.BatchMatMul()

    def construct(self, x):
        """Attention construct"""
        B, N, _ = x.shape
        q = P.Reshape()(self.q(x), (B, N, self.num_heads, self.head_dim))
        q = P.Transpose()(q, (0, 2, 1, 3))

        k = P.Reshape()(self.k(x), (B, N, self.num_heads, self.head_dim))
        k = P.Transpose()(k, (0, 2, 1, 3))
        # qk = P.Reshape()(self.qk(x), (B, N, 2, self.num_heads, self.head_dim))
        # qk = P.Transpose()(qk, (2, 0, 3, 1, 4))

        v = P.Reshape()(self.v(x), (B, N, self.num_heads, -1))
        v = P.Transpose()(v, (0, 2, 1, 3))

        attn = self.matmul(q, P.Transpose()(k, (0, 1, 3, 2))) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = P.Transpose()(self.matmul(attn, v), (0, 2, 1, 3))
        x = P.Reshape()(x, (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """ TNT Block"""

    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer((inner_dim,))
            self.inner_attn = Attention(
                inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer((inner_dim,))
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer((num_words * inner_dim,))
            self.proj = nn.Dense(in_channels=num_words * inner_dim, out_channels=outer_dim, has_bias=False)
            self.proj_norm2 = norm_layer((outer_dim,))
        # Outer
        self.outer_norm1 = norm_layer((outer_dim,))
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath1D(drop_path)
        self.outer_norm2 = norm_layer((outer_dim,))
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = 0
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)
        self.zeros = Tensor(np.zeros([1, 1, 1]), dtype=mstype.float32)

    def construct(self, inner_tokens, outer_tokens):
        """TNT Block construct"""

        if self.has_inner:
            inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))  # B*N, k*k, c
            inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))  # B*N, k*k, c
            B, N, C = P.Shape()(outer_tokens)
            zeros = P.Tile()(self.zeros, (B, 1, C))
            proj = self.proj_norm2(self.proj(self.proj_norm1(P.Reshape()(inner_tokens, (B, N - 1, -1,)))))
            proj = P.Cast()(proj, mstype.float32)
            proj = P.Concat(1)((zeros, proj))
            outer_tokens = outer_tokens + proj  # B, N, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens


class PatchEmbed(nn.Cell):
    """ Image to Visual Word Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, outer_dim=768, inner_dim=24, inner_stride=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)

        self.unfold = UnfoldKernelEqPatch(kernel_size=patch_size, strides=patch_size)
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=inner_dim, kernel_size=7, stride=inner_stride,
                              pad_mode='pad', padding=3, has_bias=False)

    def construct(self, x):
        B = x.shape[0]
        x = self.unfold(x)  # B, Ck2, N
        x = self.proj(x)  # B*N, C, 8, 8
        x = P.Reshape()(x, (B * self.num_patches, self.inner_dim, -1,))  # B*N, 8*8, C
        x = P.Transpose()(x, (0, 2, 1))
        return x


class TNT(nn.Cell):
    """ TNT (Transformer in Transformer) for computer vision
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, outer_dim=768, inner_dim=48,
                 depth=12, outer_num_heads=12, inner_num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, inner_stride=4, se=0,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.outer_dim = outer_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, outer_dim=outer_dim,
            inner_dim=inner_dim, inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words

        self.proj_norm1 = norm_layer((num_words * inner_dim,))
        self.proj = nn.Dense(in_channels=num_words * inner_dim, out_channels=outer_dim, has_bias=False)
        self.proj_norm2_tnt = norm_layer((outer_dim,))

        self.cls_token = Parameter(Tensor(trunc_array([1, 1, outer_dim]), dtype=mstype.float32), name="cls_token",
                                   requires_grad=True)
        self.outer_pos = Parameter(Tensor(trunc_array([1, num_patches + 1, outer_dim]), dtype=mstype.float32),
                                   name="outer_pos")
        self.inner_pos = Parameter(Tensor(trunc_array([1, num_words, inner_dim]), dtype=mstype.float32))
        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads,
                    inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
        self.blocks = nn.CellList(blocks)
        self.norm = norm_layer((outer_dim,))

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(outer_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        mask = np.zeros([1, num_patches + 1, 1])
        mask[:, 0] = 1
        self.mask = Tensor(mask, dtype=mstype.float32)
        self.head = nn.Dense(in_channels=outer_dim, out_channels=num_classes, has_bias=False)

        self.init_weights()
        print("================================success================================")

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def forward_features(self, x):
        """TNT forward_features"""
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 8*8, C

        outer_tokens = self.proj_norm2_tnt(
            self.proj(self.proj_norm1(P.Reshape()(inner_tokens, (B, self.num_patches, -1,)))))
        outer_tokens = P.Cast()(outer_tokens, mstype.float32)
        outer_tokens = P.Concat(1)((P.Tile()(self.cls_token, (B, 1, 1)), outer_tokens))

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)  # [batch_size, num_patch+1, outer_dim)
        return outer_tokens[:, 0]

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def tnt_s_patch16_224(args):
    """tnt_s_patch16_224"""

    patch_size = 16
    inner_stride = 4
    outer_dim = 384
    inner_dim = 24
    outer_num_heads = 6
    inner_num_heads = 4
    drop_path_rate = args.drop_path_rate
    drop_out = args.drop_out
    num_classes = args.num_classes
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=12,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, drop_path_rate=drop_path_rate, drop_out=drop_out, num_classes=num_classes)
    return model


def tnt_b_patch16_224(args):
    """tnt_b_patch16_224"""

    patch_size = 16
    inner_stride = 4
    outer_dim = 640
    inner_dim = 40
    outer_num_heads = 10
    inner_num_heads = 4
    drop_path_rate = args.drop_path_rate
    drop_out = args.drop_out
    num_classes = args.num_classes
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=12,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, drop_path_rate=drop_path_rate, drop_out=drop_out, num_classes=num_classes)
    return model
