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
import collections.abc
from itertools import repeat
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.ops import operations as P
import mindspore.common.initializer as weight_init
import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'classifier': 'head',
        **kwargs
    }


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0)  # always be 0
        # seed must be 0, if set to other value, it's not rand for multiple call
        self.rand = P.UniformReal(seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)  # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


def swish(x):
    return x * P.Sigmoid()(x)


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_features, hidden_features, 1, 1, has_bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        ])
        self.proj = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, pad_mode='pad', padding=1, group=hidden_features, has_bias=True)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features)
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(hidden_features, out_features, 1, 1, has_bias=True),
            nn.BatchNorm2d(out_features),
        ])
        self.drop = nn.Dropout(1. - drop)

    def construct(self, x, H, W):
        B, _, C = x.shape
        x = ops.Transpose()(x, (0, 2, 1)).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = ops.Transpose()(x.reshape(B, C, -1), (0, 2, 1))
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Dense(dim, self.qk_dim, has_bias=qkv_bias)
        self.k = nn.Dense(dim, self.qk_dim, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1. - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1. - proj_drop)

        self.sr_ratio = sr_ratio
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.SequentialCell([
                nn.Conv2d(dim, dim, kernel_size=sr_ratio,
                          stride=sr_ratio, group=dim, has_bias=True),
                nn.BatchNorm2d(dim),
            ])

        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              self.qk_dim // self.num_heads)
        q = ops.Transpose()(q, (0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = ops.Transpose()(x, (0, 2, 1)).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1)
            x_ = ops.Transpose()(x_, (0, 2, 1))
            k = self.k(x_).reshape(B, -1, self.num_heads,
                                   self.qk_dim // self.num_heads)
            k = ops.Transpose()(k, (0, 2, 1, 3))
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads)
            v = ops.Transpose()(v, (0, 2, 1, 3))
        else:
            k = self.k(x).reshape(B, N, self.num_heads,
                                  self.qk_dim // self.num_heads)
            k = ops.Transpose()(k, (0, 2, 1, 3))
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads)
            v = ops.Transpose()(v, (0, 2, 1, 3))

        attn = mindspore.ops.matmul(q, ops.Transpose()(
            k, (0, 1, 3, 2))) * self.scale + relative_pos

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = mindspore.ops.matmul(attn, v)
        x = ops.Transpose()(x, (0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer([dim])
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else ops.Identity()
        self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, pad_mode='pad',
                              padding=1, group=dim, has_bias=True)

    def construct(self, x, H, W, relative_pos):
        B, _, C = x.shape
        cnn_feat = ops.Transpose()(x, (0, 2, 1)).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = ops.Transpose()(x.reshape(B, C, H*W), (0, 2, 1))
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.norm = nn.LayerNorm([embed_dim])

    def construct(self, x):
        _, _, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        _B, _C, _H, _W = x.shape
        x = ops.Transpose()(x.reshape(_B, _C, _H*_W), (0, 2, 1))
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class CMT(nn.Cell):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=None, stem_channel=16,
                 fc_dim=1280, num_heads=None, mlp_ratios=None, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 depths=None, qk_ratio=1, sr_ratios=None, dp=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or nn.LayerNorm

        self.stem_conv1 = nn.Conv2d(
            3, stem_channel, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel)

        self.stem_conv2 = nn.Conv2d(
            stem_channel, stem_channel, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel)

        self.stem_conv3 = nn.Conv2d(
            stem_channel, stem_channel, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size//2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size//4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size//8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size//16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_a = P.Zeros()(
            (num_heads[0], self.patch_embed_a.num_patches, self.patch_embed_a.num_patches//sr_ratios[0]//sr_ratios[0]),
            mindspore.float32)
        self.relative_pos_a = Parameter(self.relative_pos_a)
        self.relative_pos_b = P.Zeros()(
            (num_heads[1], self.patch_embed_b.num_patches, self.patch_embed_b.num_patches//sr_ratios[1]//sr_ratios[1]),
            mindspore.float32)
        self.relative_pos_b = Parameter(self.relative_pos_b)
        self.relative_pos_c = P.Zeros()(
            (num_heads[2], self.patch_embed_c.num_patches, self.patch_embed_c.num_patches//sr_ratios[2]//sr_ratios[2]),
            mindspore.float32)
        self.relative_pos_c = Parameter(self.relative_pos_c)
        self.relative_pos_d = P.Zeros()(
            (num_heads[3], self.patch_embed_d.num_patches, self.patch_embed_d.num_patches//sr_ratios[3]//sr_ratios[3]),
            mindspore.float32)
        self.relative_pos_d = Parameter(self.relative_pos_d)

        # stochastic depth decay rule
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks_a = nn.CellList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                    cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.CellList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                    cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.CellList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                    cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.CellList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                    cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # Classifier head
        self._fc = nn.Conv2d(
            embed_dims[-1], fc_dim, kernel_size=1, has_bias=True)
        self._bn = nn.BatchNorm2d(fc_dim)
        self._drop = nn.Dropout(1. - dp)
        self.head = nn.Dense(
            fc_dim, num_classes) if num_classes > 0 else ops.Identity()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (nn.LayerNorm, nn.BatchNorm2d)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)
        for _, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)

        x = ops.Transpose()(x.reshape(B, H, W, -1), (0, 3, 1, 2))
        x, (H, W) = self.patch_embed_b(x)
        for _, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)

        x = ops.Transpose()(x.reshape(B, H, W, -1), (0, 3, 1, 2))
        x, (H, W) = self.patch_embed_c(x)
        for _, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)

        x = ops.Transpose()(x.reshape(B, H, W, -1), (0, 3, 1, 2))
        x, (H, W) = self.patch_embed_d(x)
        for _, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)

        B, _, C = x.shape

        x = self._fc(ops.Transpose()(x, (0, 2, 1)).reshape(B, C, H, W))
        x = self._bn(x)
        x = swish(x)
        x = ops.AdaptiveAvgPool2D(output_size=1)(x).squeeze(2).squeeze(2)
        x = self._drop(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_cmt_model(pretrained=False, distilled=False, **kwargs):
    default_cfg = _cfg()
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)

    model = CMT(img_size=img_size, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfg

    return model


def cmt_ti(pretrained=False, **kwargs):
    """
    CMT-Tiny
    """
    model_kwargs = dict(
        embed_dims=[46, 92, 184, 368], num_heads=[1, 2, 4, 8], mlp_ratios=[3.6, 3.6, 3.6, 3.6],
        depths=[2, 2, 10, 2], sr_ratios=[8, 4, 2, 1], qkv_bias=True, **kwargs)
    model = _create_cmt_model(pretrained=pretrained, **model_kwargs)
    return model


def cmt_xs(pretrained=False, **kwargs):
    """
    CMT-XS: dim x 0.9, depth x 0.8, input 192
    """
    model_kwargs = dict(
        qkv_bias=True, embed_dims=[52, 104, 208, 416], stem_channel=16, num_heads=[1, 2, 4, 8],
        depths=[3, 3, 12, 3], mlp_ratios=[3.77, 3.77, 3.77, 3.77], qk_ratio=1, sr_ratios=[8, 4, 2, 1], **kwargs)
    model = _create_cmt_model(pretrained=pretrained, **model_kwargs)
    return model


def cmt_s(pretrained=False, **kwargs):
    """
    CMT-Small
    """
    model_kwargs = dict(
        qkv_bias=True, embed_dims=[64, 128, 256, 512], stem_channel=32, num_heads=[1, 2, 4, 8],
        depths=[3, 3, 16, 3], mlp_ratios=[4, 4, 4, 4], qk_ratio=1, sr_ratios=[8, 4, 2, 1], **kwargs)
    model = _create_cmt_model(pretrained=pretrained, **model_kwargs)
    return model


def cmt_b(pretrained=False, **kwargs):
    """
    CMT-Base
    """
    model_kwargs = dict(
        qkv_bias=True, embed_dims=[76, 152, 304, 608], stem_channel=38, num_heads=[1, 2, 4, 8],
        depths=[4, 4, 20, 4], mlp_ratios=[4, 4, 4, 4], qk_ratio=1, sr_ratios=[8, 4, 2, 1], dp=0.3, **kwargs)
    model = _create_cmt_model(pretrained=pretrained, **model_kwargs)
    return model
