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

# Part of this file was copied from https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp

import os
from itertools import repeat
import collections.abc

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
import mindspore.numpy as mnp


from quan_conv import QuanConv


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        self.rand = P.UniformReal(seed=0)
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


def _cfg(url="", crop_pct=0.96):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": crop_pct,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "classifier": "head",
    }


default_cfgs = {
    "wave_T": _cfg(crop_pct=0.9),
    "wave_S": _cfg(crop_pct=0.9),
    "wave_M": _cfg(crop_pct=0.9),
    "wave_B": _cfg(crop_pct=0.875),
}


class LearnableBias(nn.Cell):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = mindspore.Parameter(
            ops.Zeros()((1, out_chn, 1, 1), mindspore.float32), requires_grad=True
        )

    def construct(self, x):
        out = x + self.bias.expand_as(x)
        return out


class prelu(nn.Cell):
    def __init__(self, out_chn):
        super(prelu, self).__init__()
        self.w = mindspore.Parameter(
            0.25 * ops.Ones()((1, out_chn, 1, 1), mindspore.float32), requires_grad=True
        )

    def construct(self, x):
        me_max_Tensor = x.copy()
        me_max_Tensor[me_max_Tensor < 0] = 0
        me_min_Tensor = x.copy()
        me_min_Tensor[me_min_Tensor > 0] = 0

        x = ops.Add()(me_max_Tensor, ops.Mul()(self.w, me_min_Tensor))
        return x


class RPReLU(nn.Cell):
    def __init__(self, out_chn):
        super(RPReLU, self).__init__()
        self.move1 = LearnableBias(out_chn)
        self.move2 = LearnableBias(out_chn)

        self.act = prelu(out_chn)

    def construct(self, x):
        x = self.move1(x)

        x = self.act(x)
        x = self.move2(x)
        return x


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.move1 = LearnableBias(in_features)
        self.fc1 = QuanConv(
            in_features,
            hidden_features,
            kernel_size=1,
            quan_name_w="xnor",
            quan_name_a="xnor",
            nbit_w=1,
            nbit_a=1,
            bias=False,
        )
        self.act1 = RPReLU(hidden_features)
        self.act2 = RPReLU(hidden_features)
        self.act3 = RPReLU(out_features)
        self.move2 = LearnableBias(hidden_features)
        self.fc2 = QuanConv(
            hidden_features,
            out_features,
            kernel_size=1,
            quan_name_w="xnor",
            quan_name_a="xnor",
            nbit_w=1,
            nbit_a=1,
            bias=False,
        )
        self.drop = nn.Dropout(p=drop)
        self.norm1 = nn.BatchNorm2d(num_features=hidden_features)
        self.norm2 = nn.BatchNorm2d(num_features=out_features)
        self.norm0 = nn.BatchNorm2d(num_features=hidden_features)

        self.hidden_features = hidden_features
        self.in_features = in_features
        self.out_features = out_features

    def construct(self, x):
        if self.hidden_features // self.in_features == 4:
            inp = mnp.tile(x, (1, 4, 1, 1))
        else:
            inp = (
                x[:, ::4, :, :] + x[:, 1::4, :, :] +
                x[:, 2::4, :, :] + x[:, 3::4, :, :]
            ) / 4.0
        inp = self.norm0(inp)
        inp = self.act1(inp)
        x = self.move1(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = x + inp
        x = self.act2(x)
        x = self.drop(x)

        if self.out_features // self.hidden_features == 12:
            inp = mnp.tile(x, (1, 12, 1, 1))
        else:
            inp = (
                x[:, ::4, :, :] + x[:, 1::4, :, :] +
                x[:, 2::4, :, :] + x[:, 3::4, :, :]
            ) / 4.0
        x = self.move2(x)
        x2 = self.fc2(x)
        x2 = self.norm2(x2)
        x2 = x2 + inp
        x2 = self.act3(x2)
        x2 = self.drop(x2)
        return x2


class PATM(nn.Cell):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super(PATM, self).__init__()

        self.fc_h = QuanConv(dim, dim, kernel_size=1, quan_name_w='xnor', quan_name_a='xnor',
                             nbit_w=1, nbit_a=1, bias=False)
        self.fc_w = QuanConv(dim, dim, kernel_size=1, quan_name_w='xnor', quan_name_a='xnor',
                             nbit_w=1, nbit_a=1, bias=False)
        self.fc_c = QuanConv(dim, dim, kernel_size=1, quan_name_w='xnor', quan_name_a='xnor',
                             nbit_w=1, nbit_a=1, bias=False)
        self.move_fch = LearnableBias(dim)
        self.move_fcw = LearnableBias(dim)
        self.move_fcc = LearnableBias(dim)
        self.fc_h_bn = nn.BatchNorm2d(num_features=dim)
        self.fc_w_bn = nn.BatchNorm2d(num_features=dim)
        self.fc_h_act = RPReLU(dim)
        self.fc_w_act = RPReLU(dim)

        self.tfc_h = QuanConv(
            2 * dim,
            dim,
            kernel_size=(1, 7),
            stride=1,
            padding=(0, 0, 7 // 2, 7 // 2),
            group=dim,
            quan_name_w="xnor",
            quan_name_a="xnor",
            nbit_w=1,
            nbit_a=1,
            bias=False,
        )
        self.tfc_w = QuanConv(
            2 * dim,
            dim,
            kernel_size=(7, 1),
            stride=1,
            padding=(7 // 2, 7 // 2, 0, 0),
            group=dim,
            quan_name_w="xnor",
            quan_name_a="xnor",
            nbit_w=1,
            nbit_a=1,
            bias=False,
        )
        self.move_tfch = LearnableBias(2 * dim)
        self.move_tfcw = LearnableBias(2 * dim)

        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = QuanConv(dim, dim, kernel_size=1, quan_name_w='xnor', quan_name_a='xnor',
                             nbit_w=1, nbit_a=1, bias=False)
        self.move_proj = LearnableBias(dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.mode = mode

        if mode == "fc":
            self.move_thetah = LearnableBias(dim)
            self.theta_h_conv = QuanConv(dim, dim, kernel_size=1, quan_name_w='xnor', quan_name_a='xnor',
                                         nbit_w=1, nbit_a=1, bias=False)
            self.theta_h_bn = nn.BatchNorm2d(dim)
            self.theta_h_act = RPReLU(dim)
            self.move_thetaw = LearnableBias(dim)
            self.theta_w_conv = QuanConv(dim, dim, kernel_size=1, quan_name_w='xnor', quan_name_a='xnor',
                                         nbit_w=1, nbit_a=1, bias=False)
            self.theta_w_bn = nn.BatchNorm2d(dim)
            self.theta_w_act = RPReLU(dim)
        else:
            self.theta_h_conv = nn.SequentialCell(
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    stride=1,
                    pad_mode="pad",
                    padding=1,
                    group=dim,
                    bias=False,
                ),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )
            self.theta_w_conv = nn.SequentialCell(
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    stride=1,
                    pad_mode="pad",
                    padding=1,
                    group=dim,
                    bias=False,
                ),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )

        self.act_c = RPReLU(dim)
        self.act_h = RPReLU(dim)
        self.act_w = RPReLU(dim)
        self.act_proj = RPReLU(dim)
        self.norm_c = nn.BatchNorm2d(dim)
        self.norm_h = nn.BatchNorm2d(dim)
        self.norm_w = nn.BatchNorm2d(dim)
        self.norm_proj = nn.BatchNorm2d(dim)

        self.reshape = ops.Reshape()
        self.permute = ops.Transpose()
        self.unqueeze = ops.ExpandDims()

    def construct(self, inp, round_num=-1):

        B, C, _, _ = inp.shape
        theta_h_in = self.move_thetah(inp)

        theta_h = self.theta_h_conv(theta_h_in)

        theta_h = self.theta_h_bn(theta_h)

        theta_h = self.theta_h_act(theta_h + theta_h_in)

        theta_w_in = self.move_thetaw(inp)
        theta_w = self.theta_w_conv(theta_w_in)
        theta_w = self.theta_w_bn(theta_w)
        theta_w = self.theta_w_act(theta_w + theta_w_in)

        x_h_in = self.move_fch(inp)
        x_h = self.fc_h(x_h_in)
        x_h = self.fc_h_bn(x_h)
        x_h = self.fc_h_act(x_h + x_h_in)

        x_w_in = self.move_fcw(inp)

        x_w = self.fc_w(x_w_in)

        x_w = self.fc_w_bn(x_w)
        x_w = self.fc_w_act(x_w + x_w_in)

        x_h = ops.Concat(axis=1)(
            [x_h * ops.cos(theta_h), x_h * ops.sin(theta_h)])

        x_w = ops.Concat(axis=1)(
            [x_w * ops.cos(theta_w), x_w * ops.sin(theta_w)])

        x_h = self.move_tfch(x_h)
        x_w = self.move_tfcw(x_w)

        fc_c_in = self.move_fcc(inp)
        th = x_h[:, ::2, :, :] + x_h[:, 1::2, :, :]
        tw = x_w[:, ::2, :, :] + x_w[:, 1::2, :, :]

        ah = self.norm_h(self.tfc_h(x_h)) + (th / 2.0)
        aw = self.norm_w(self.tfc_w(x_w)) + (tw / 2.0)

        h = self.act_h(ah)
        w = self.act_w(aw)
        c = self.act_c(self.norm_c(self.fc_c(fc_c_in)) + fc_c_in)
        a = ops.AdaptiveAvgPool2D(output_size=(1, 1))(h + w + c + inp)

        a = ops.expand_dims(
            ops.expand_dims(
                ops.Softmax(axis=0)(
                    ops.Transpose()(self.reweight(a).reshape(B, C, 3), (2, 0, 1))
                ),
                axis=-1,
            ),
            axis=-1,
        )
        x = h * a[0] + w * a[1] + c * a[2] + inp
        x_in = self.move_proj(x)
        x = self.act_proj(self.norm_proj(self.proj(x_in)) + x_in)
        x = self.proj_drop(x)
        return x


class WaveBlock(nn.Cell):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(
            dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else ops.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

        self.move_h = LearnableBias(dim)
        self.move_w = LearnableBias(dim)
        self.sfc_h = QuanConv(
            dim,
            dim,
            kernel_size=(1, 7),
            stride=1,
            pad_mode="pad",
            padding=(0, 0, 7 // 2, 7 // 2),
            group=dim,
            quan_name_w="xnor",
            quan_name_a="xnor",
            nbit_w=1,
            nbit_a=1,
            bias=False,
        )
        self.sfc_w = QuanConv(
            dim,
            dim,
            kernel_size=(7, 1),
            stride=1,
            pad_mode="pad",
            padding=(7 // 2, 7 // 2, 0, 0),
            group=dim,
            quan_name_w="xnor",
            quan_name_a="xnor",
            nbit_w=1,
            nbit_a=1,
            bias=False,
        )
        self.act1 = RPReLU(dim)
        self.act2 = RPReLU(dim)
        self.norm_h = nn.BatchNorm2d(dim)
        self.norm_w = nn.BatchNorm2d(dim)

        self.weight = mindspore.Parameter(ops.Zeros()(2, mindspore.float32))

    def construct(self, x, round_num=-1):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        mid = self.norm2(x)
        a = self.mlp(mid)
        mid_h = self.move_h(mid)
        mid_w = self.move_w(mid)
        b = self.act1(self.norm_h(self.sfc_h(mid_h)) + mid_h)
        c = self.act2(self.norm_w(self.sfc_w(mid_w)) + mid_w)
        x = x + self.drop_path(
            self.weight[0] * b
            + self.weight[1] * c
            + (1 - self.weight[0] - self.weight[1]) * a
        )
        return x


class PatchEmbedOverlapping(nn.Cell):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768,
                 norm_layer=nn.BatchNorm2d, group=1, use_norm=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.patch_size = patch_size

        if isinstance(padding, tuple) and len(padding) == 2:
            print("strange padding: ", padding)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            group=group,
            has_bias=True,
        )
        self.norm = norm_layer(
            embed_dim) if use_norm else ops.Identity()
        self.act = RPReLU(embed_dim)

    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, norm_layer=nn.BatchNorm2d, use_norm=True):
        super().__init__()
        assert patch_size == 2, patch_size
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, pad_mode="same")
        self.pool3 = nn.MaxPool2d(kernel_size=7, stride=2, pad_mode="same")
        self.proj = nn.Conv2d(
            in_embed_dim,
            out_embed_dim,
            kernel_size=(1, 1),
            stride=1,
            pad_mode="pad",
            padding=0,
            has_bias=True,
        )  # must add has_bias
        self.norm = norm_layer(
            out_embed_dim) if use_norm else ops.Identity()
        self.act = RPReLU(out_embed_dim)

    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = (self.pool0(x) + self.pool1(x) +
             self.pool2(x) + self.pool3(x)) / 4.0
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0., norm_layer=nn.BatchNorm2d, mode='fc', **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate *
            (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            WaveBlock(
                dim,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                drop_path=block_dpr,
                norm_layer=norm_layer,
                mode=mode,
            )
        )

    blocks = nn.SequentialCell(*blocks)
    return blocks


class WaveNet(nn.Cell):
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dims=None, transitions=None, mlp_ratios=None,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.BatchNorm2d, fork_feat=False, mode='fc', ds_use_norm=True, args=None):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbedOverlapping(
            patch_size=7,
            stride=4,
            padding=2,
            in_chans=3,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer,
            use_norm=ds_use_norm,
        )

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                mode=mode,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(
                    Downsample(
                        embed_dims[i],
                        embed_dims[i + 1],
                        patch_size,
                        norm_layer=norm_layer,
                        use_norm=ds_use_norm,
                    )
                )

        self.network = nn.CellList(network)

        if self.fork_feat:

            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    layer = ops.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = (
                nn.Dense(embed_dims[-1], num_classes)
                if num_classes > 0
                else ops.Identity()
            )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Dense(self.embed_dim,
                     num_classes) if num_classes > 0 else ops.Identity()
        )

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []

        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def construct(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        cls_out = self.head(ops.Squeeze()(
            ops.AdaptiveAvgPool2D(output_size=1)(x)))
        return cls_out


def MyNorm(dim):
    return nn.GroupNorm(1, dim)


def BiMLP_S(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(
        layers,
        embed_dims=embed_dims,
        patch_size=7,
        transitions=transitions,
        mlp_ratios=mlp_ratios,
        **kwargs,
    )
    model.default_cfg = default_cfgs["wave_T"]
    return model


def BiMLP_M(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(
        layers,
        embed_dims=embed_dims,
        patch_size=7,
        transitions=transitions,
        mlp_ratios=mlp_ratios,
        norm_layer=MyNorm,
        **kwargs,
    )
    model.default_cfg = default_cfgs["wave_S"]
    return model
