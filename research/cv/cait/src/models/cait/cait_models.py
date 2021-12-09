# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore.common.initializer as weight_init
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

from src.models.cait.misc import to_2tuple, Identity, DropPath1D


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Dense(in_chans * patch_size[0] * patch_size[1], embed_dim, has_bias=False)
        self.norm = norm_layer((embed_dim,), epsilon=1e-8) if norm_layer else Identity()

    def construct(self, x):
        B, C, H, W = x.shape
        x = ops.Reshape()(x, (B, C, H // self.patch_size[0], self.patch_size[0], W // self.patch_size[1],
                              self.patch_size[1]))
        x = ops.Transpose()(x, (0, 2, 4, 1, 3, 5))
        x = ops.Reshape()(x, (B, self.num_patches, -1))
        x = self.proj(x)
        x = self.norm(x)
        return x


class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=False)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=False)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Class_Attention(nn.Cell):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        B, N, C = x.shape
        q = P.Reshape()(self.q(x[:, 0]), (B, 1, self.num_heads, C // self.num_heads))
        q = P.Transpose()(q, (0, 2, 1, 3))
        k = P.Reshape()(self.k(x), (B, N, self.num_heads, C // self.num_heads))
        k = P.Transpose()(k, (0, 2, 3, 1))
        q = q * self.scale
        v = P.Reshape()(self.v(x), (B, N, self.num_heads, C // self.num_heads))
        v = P.Transpose()(v, (0, 2, 1, 3))

        attn = ops.BatchMatMul()(q, k)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x_cls = ops.BatchMatMul()(attn, v)
        x_cls = P.Transpose()(x_cls, (0, 2, 1, 3))
        x_cls = P.Reshape()(x_cls, (B, 1, C))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class LayerScale_Block_CA(nn.Cell):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Class_Attention,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer((dim,), epsilon=1e-8)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,), epsilon=1e-8)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = Parameter(Tensor(init_values * np.ones([1, 1, dim]), mstype.float32))
        self.gamma_2 = Parameter(Tensor(init_values * np.ones([1, 1, dim]), mstype.float32))

    def construct(self, x, x_cls):
        u = P.Concat(1)((x_cls, x))
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))

        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class Attention_talking_head(nn.Cell):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)

        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=False)

        self.proj_l = nn.Dense(in_channels=num_heads, out_channels=num_heads, has_bias=False)
        self.proj_w = nn.Dense(in_channels=num_heads, out_channels=num_heads, has_bias=False)

        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        B_, N, C = x.shape
        q = ops.Reshape()(self.q(x), (B_, N, self.num_heads, C // self.num_heads))
        q = ops.Transpose()(q, (0, 2, 1, 3)) * self.scale
        k = ops.Reshape()(self.k(x), (B_, N, self.num_heads, C // self.num_heads))
        k = ops.Transpose()(k, (0, 2, 3, 1))
        v = ops.Reshape()(self.v(x), (B_, N, self.num_heads, C // self.num_heads))
        v = ops.Transpose()(v, (0, 2, 1, 3))
        attn = ops.BatchMatMul()(q, k)
        attn = P.Transpose()(self.proj_l(P.Transpose()(attn, (0, 2, 3, 1,))), (0, 3, 1, 2))
        attn = self.softmax(attn)
        attn = P.Transpose()(self.proj_w(P.Transpose()(attn, (0, 2, 3, 1,))), (0, 3, 1, 2))
        attn = self.attn_drop(attn)
        x = P.Transpose()(ops.BatchMatMul()(attn, v), (0, 2, 1, 3))
        x = P.Reshape()(x, (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale_Block(nn.Cell):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention_talking_head,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer((dim,), epsilon=1e-8)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,), epsilon=1e-8)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = Parameter(Tensor(init_values * np.ones([1, 1, dim]), mstype.float32))
        self.gamma_2 = Parameter(Tensor(init_values * np.ones([1, 1, dim]), mstype.float32))

    def construct(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class cait_models(nn.Cell):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(self, image_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 block_layers=LayerScale_Block,
                 block_layers_token=LayerScale_Block_CA,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=Attention_talking_head, Mlp_block=Mlp,
                 init_scale=1e-4,
                 Attention_block_token_only=Class_Attention,
                 Mlp_block_token_only=Mlp,
                 depth_token_only=2,
                 mlp_ratio_clstk=4.0):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(Tensor(np.zeros([1, 1, embed_dim]), mstype.float32))
        self.pos_embed = Parameter(Tensor(np.zeros([1, num_patches, embed_dim]), mstype.float32))
        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        dpr = [drop_path_rate for _ in range(depth)]
        self.blocks = nn.CellList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.blocks_token_only = nn.CellList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, init_values=init_scale)
            for _ in range(depth_token_only)])

        self.norm = norm_layer((embed_dim,), epsilon=1e-8)
        channel_mask = np.zeros([1, num_patches + 1, 1])
        channel_mask[0] = 1
        self.channel_mask = Tensor(channel_mask, mstype.float32)

        self.head = nn.Dense(in_channels=embed_dim, out_channels=num_classes, has_bias=False) if num_classes > 0 else \
            Identity()

        self.pos_embed.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                        self.pos_embed.shape,
                                                        self.pos_embed.dtype))
        self.cls_token.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                        self.cls_token.shape,
                                                        self.cls_token.dtype))
        self.init_weights()

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
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = P.Tile()(self.cls_token, (B, 1, 1))
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        for blk in self.blocks_token_only:
            cls_tokens = blk(x, cls_tokens)
        x = P.Concat(1)((cls_tokens, x))

        x = self.norm(x)
        return x[:, 0]

    def construct(self, x):
        x = self.forward_features(x)

        x = self.head(x)

        return x


def cait_XXS24_224(args):
    num_classes = args.num_classes
    drop_path_rate = args.drop_path_rate
    image_size = args.image_size
    assert image_size == 224
    model = cait_models(
        image_size=image_size, patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=False,
        norm_layer=nn.LayerNorm,
        init_scale=1e-5,
        depth_token_only=2, num_classes=num_classes, drop_path_rate=drop_path_rate)

    return model
