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
from functools import partial
from itertools import repeat

import mindspore
import mindspore.common.initializer as weight_init
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P

import cfg

from .modules import LayerNormParallel, ModuleParallel, TokenExchange


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0)  # always be 0
        self.rand = P.UniformReal(
            seed=seed
        )  # seed must be 0, if set to other value, it's not rand for multiple call
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


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Dense(in_features, hidden_features))
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(nn.GELU(False))
        self.fc2 = ModuleParallel(nn.Dense(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(p=1 - drop))

    def construct(self, x, H, W):
        x = self.fc1(x)
        x = [self.dwconv(x[0], H, W), self.dwconv(x[1], H, W)]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=1.0,
            proj_drop=1.0,
            sr_ratio=1,
        ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = ModuleParallel(nn.Dense(dim, dim, has_bias=qkv_bias))
        self.kv = ModuleParallel(nn.Dense(dim, dim * 2, has_bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(p=1 - attn_drop))
        self.proj = ModuleParallel(nn.Dense(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(p=1 - proj_drop))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ModuleParallel(
                nn.Conv2d(
                    dim, dim, kernel_size=sr_ratio, stride=sr_ratio, has_bias=True
                )
            )
            self.norm = LayerNormParallel(dim)
        self.exchange = TokenExchange()
        self.softmax = nn.Softmax(axis=-1)
        self.expand = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.matmul = ops.BatchMatMul()

    def construct(self, x, H, W, mask):
        B, N, C = x[0].shape
        q = self.q(x)
        q = [
            self.transpose(
                q_.reshape(B, N, self.num_heads, C // self.num_heads), (0, 2, 1, 3)
            )
            for q_ in q
        ]

        if self.sr_ratio > 1:
            tmp = [self.transpose(x_, (0, 2, 1)).reshape(B, C, H, W) for x_ in x]
            tmp = self.sr(tmp)
            tmp = [self.transpose(tmp_.reshape(B, C, -1), (0, 2, 1)) for tmp_ in tmp]
            kv = self.kv(self.norm(tmp))
        else:
            kv = self.kv(x)
        kv = [
            self.transpose(
                kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads),
                (2, 0, 3, 1, 4),
            )
            for kv_ in kv
        ]
        k, v = [kv_[0] for kv_ in kv], [kv_[1] for kv_ in kv]

        attn = [
            (self.matmul(q_, self.transpose(k_, (0, 1, 3, 2)))) * self.scale
            for (q_, k_) in zip(q, k)
        ]
        attn = [self.softmax(attn_) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [
            self.transpose((self.matmul(attn_, v_)), (0, 2, 1, 3)).reshape(B, N, C)
            for (attn_, v_) in zip(attn, v)
        ]
        x = self.proj(x)
        x = self.proj_drop(x)

        if mask is not None:
            x = [x_ * self.expand(mask_, 2) for (x_, mask_) in zip(x, mask)]
            x = self.exchange(x, mask, mask_threshold=0.02)

        return x


class DWConv(nn.Cell):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            3,
            1,
            padding=(1, 1, 1, 1),
            has_bias=True,
            group=dim,
            pad_mode="pad",
        )
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x, H, W):
        B, N, C = x.shape
        x = self.transpose(x, (0, 2, 1)).view(B, C, H, W)
        x = self.dwconv(x)
        B, N, C, _ = x.shape
        x = self.transpose(self.reshape(x, (B, N, -1)), (0, 2, 1))

        return x


class Block(nn.Cell):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=1.0,
            attn_drop=1.0,
            drop_path=0.0,
            sr_ratio=1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        ):
        super().__init__()
        self.norm1 = LayerNormParallel(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            ModuleParallel(DropPath(drop_path))
            if drop_path > 0.0
            else ModuleParallel(ops.Identity())
        )
        self.norm2 = LayerNormParallel(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def construct(self, x, H, W, mask=None):
        out = self.drop_path(self.attn(self.norm1(x), H, W, mask=mask))
        x = [x_ + out_ for (x_, out_) in zip(x, out)]
        out = self.drop_path(self.mlp(self.norm2(x), H, W))
        x = [x_ + out_ for (x_, out_) in zip(x, out)]
        return x


class OverlapPatchEmbed(nn.Cell):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = ModuleParallel(
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=(
                    patch_size[0] // 2,
                    patch_size[1] // 2,
                    patch_size[0] // 2,
                    patch_size[1] // 2,
                ),
                pad_mode="pad",
                has_bias=True,
            )
        )
        self.norm = LayerNormParallel(embed_dim)
        self.embed_dim = embed_dim
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.proj(x)
        B, _, H, W = x[0].shape
        x = [
            self.transpose(self.reshape(x_, (B, self.embed_dim, -1)), (0, 2, 1))
            for x_ in x
        ]
        x = self.norm(x)
        return x, H, W


class PredictorLG(nn.Cell):
    """Image to Patch Embedding from DydamicVit"""

    def __init__(self, embed_dim=384):
        super().__init__()
        self.score_nets = nn.CellList(
            [
                nn.SequentialCell(
                    nn.LayerNorm([embed_dim]),
                    nn.Dense(embed_dim, embed_dim),
                    nn.GELU(False),
                    nn.Dense(embed_dim, embed_dim // 2),
                    nn.GELU(False),
                    nn.Dense(embed_dim // 2, embed_dim // 4),
                    nn.GELU(False),
                    nn.Dense(embed_dim // 4, 2),
                    nn.LogSoftmax(axis=-1),
                )
                for _ in range(cfg.num_parallel)
            ]
        )

    def construct(self, x):
        x = [self.score_nets[i](x[i]) for i in range(cfg.num_parallel)]
        return x


class MixVisionTransformer(nn.Cell):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dims=None,
            num_heads=None,
            mlp_ratios=None,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=1.0,
            attn_drop_rate=1.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=None,
            sr_ratios=None,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        predictor_list = [PredictorLG(embed_dims[i]) for i in range(len(depths))]
        self.score_predictor = nn.CellList(predictor_list)

        # transformer encoder
        dpr = ops.LinSpace()(
            mindspore.Tensor(0, mindspore.float32),
            mindspore.Tensor(drop_path_rate, mindspore.float32),
            sum(depths),
        )  # stochastic depth decay rule

        cur = 0
        self.block1 = nn.CellList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = LayerNormParallel(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.CellList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = LayerNormParallel(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.CellList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = LayerNormParallel(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.CellList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = LayerNormParallel(embed_dims[3])
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.softmax = nn.Softmax(axis=2)
        self.init_weights()

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    weight_init.initializer(
                        weight_init.TruncatedNormal(sigma=0.02),
                        cell.weight.shape,
                        cell.weight.dtype,
                    )
                )
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer(
                            weight_init.Zero(), cell.bias.shape, cell.bias.dtype
                        )
                    )
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(
                    weight_init.initializer(
                        weight_init.One(), cell.gamma.shape, cell.gamma.dtype
                    )
                )
                cell.beta.set_data(
                    weight_init.initializer(
                        weight_init.Zero(), cell.beta.shape, cell.beta.dtype
                    )
                )
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    weight_init.initializer("XavierUniform", cell.weight.shape)
                )
                if cell.has_bias is True:
                    cell.bias.set_data(
                        weight_init.initializer("zeros", cell.bias.shape)
                    )

    def forward_features(self, x):
        B = x[0].shape[0]
        outs = []

        masks = []
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            score = self.score_predictor[0](x)
            mask = [self.softmax(score_.reshape(B, -1, 2))[:, :, 0] for score_ in score]
            masks.append([mask_.flatten() for mask_ in mask])
            x = blk(x, H, W, mask)
        x = self.norm1(x)
        x = [self.transpose(x_.reshape(B, H, W, -1), (0, 3, 1, 2)) for x_ in x]
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            score = self.score_predictor[1](x)
            mask = [
                self.softmax(score_.reshape(B, -1, 2))[:, :, 0] for score_ in score
            ]  # mask_: [B, N]
            masks.append([mask_.flatten() for mask_ in mask])
            x = blk(x, H, W, mask)
        x = self.norm2(x)
        x = [self.transpose(x_.reshape(B, H, W, -1), (0, 3, 1, 2)) for x_ in x]
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            score = self.score_predictor[2](x)
            mask = [
                self.softmax(score_.reshape(B, -1, 2))[:, :, 0] for score_ in score
            ]  # mask_: [B, N]
            masks.append([mask_.flatten() for mask_ in mask])
            x = blk(x, H, W, mask)
        x = self.norm3(x)
        x = [self.transpose(x_.reshape(B, H, W, -1), (0, 3, 1, 2)) for x_ in x]
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            score = self.score_predictor[3](x)
            mask = [
                self.softmax(score_.reshape(B, -1, 2))[:, :, 0] for score_ in score
            ]  # mask_: [B, N]
            masks.append([mask_.flatten() for mask_ in mask])
            x = blk(x, H, W, mask)
        x = self.norm4(x)
        x = [self.transpose(x_.reshape(B, H, W, -1), (0, 3, 1, 2)) for x_ in x]
        outs.append(x)

        parallel_outs = []
        for i in range(len(x)):
            lvl_out = []
            for k in range(len(outs)):
                lvl_out.append(outs[k][i])
            parallel_outs.append(lvl_out)
        return parallel_outs, masks

    def construct(self, x):
        x, masks = self.forward_features(x)
        return x, masks


class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=1.0,
            drop_path_rate=0.9,
        )


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=1.0,
            drop_path_rate=0.9,
        )


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=1.0,
            drop_path_rate=0.9,
        )


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=1.0,
            drop_path_rate=0.9,
        )


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=1.0,
            drop_path_rate=0.9,
        )


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=1.0,
            drop_path_rate=0.9,
        )
