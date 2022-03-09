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
import os
from itertools import repeat
import collections.abc
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
import mindspore.common.initializer as weight_init


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        self.rand = P.UniformReal(seed=0)  # seed must be 0, if set to other value, it's not rand for multiple call
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


def _cfg(url='', crop_pct=.96):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': crop_pct, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'classifier': 'head'
    }


default_cfgs = {
    'wave_T': _cfg(crop_pct=0.9),
    'wave_S': _cfg(crop_pct=0.9),
    'wave_M': _cfg(crop_pct=0.9),
    'wave_B': _cfg(crop_pct=0.875),
}


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(1. - drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, has_bias=True)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, has_bias=True)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PATM(nn.Cell):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, has_bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, has_bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, has_bias=qkv_bias)
        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 0, 7 // 2, 7 // 2), group=dim,
                               has_bias=False, pad_mode='pad')
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 7 // 2, 0, 0), group=dim,
                               has_bias=False, pad_mode='pad')
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1, has_bias=True)
        self.proj_drop = nn.Dropout(1. - proj_drop)
        self.mode = mode

        if mode == 'fc':
            self.theta_h_conv = nn.SequentialCell(nn.Conv2d(dim, dim, 1, 1, has_bias=True), nn.BatchNorm2d(dim),
                                                  nn.ReLU())
            self.theta_w_conv = nn.SequentialCell(nn.Conv2d(dim, dim, 1, 1, has_bias=True), nn.BatchNorm2d(dim),
                                                  nn.ReLU())
        else:
            self.theta_h_conv = nn.SequentialCell(
                nn.Conv2d(dim, dim, 3, stride=1, padding=1, group=dim, has_bias=False),
                nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.SequentialCell(
                nn.Conv2d(dim, dim, 3, stride=1, padding=1, group=dim, has_bias=False),
                nn.BatchNorm2d(dim), nn.ReLU())

    def construct(self, x):

        B, C, _, _ = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = ops.Concat(axis=1)((x_h * (ops.Cos()(theta_h)), x_h * (ops.Sin()(theta_h))))
        x_w = ops.Concat(axis=1)((x_w * (ops.Cos()(theta_w)), x_w * (ops.Sin()(theta_w))))
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = ops.AdaptiveAvgPool2D(output_size=(1, 1))(h + w + c)
        a = ops.ExpandDims()(
            ops.ExpandDims()(ops.Softmax(axis=0)(ops.Transpose()(self.reweight(a).reshape(B, C, 3), (2, 0, 1))), -1),
            -1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WaveBlock(nn.Cell):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else ops.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlapping(nn.Cell):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 groups=1, use_norm=True):
        super(PatchEmbedOverlapping).__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(padding, padding, padding, padding),
                              group=groups, pad_mode='pad', has_bias=True)
        self.norm = norm_layer(embed_dim) if use_norm else ops.Identity()

    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, norm_layer=nn.BatchNorm2d, use_norm=True):
        super(Downsample).__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1, pad_mode='pad',
                              has_bias=True)
        self.norm = norm_layer(out_embed_dim) if use_norm else ops.Identity()

    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0., norm_layer=nn.BatchNorm2d, mode='fc', **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(WaveBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop=attn_drop, drop_path=block_dpr, norm_layer=norm_layer, mode=mode))
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

        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0],
                                                 norm_layer=norm_layer, use_norm=ds_use_norm)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, mode=mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size, norm_layer=norm_layer,
                                          use_norm=ds_use_norm))

        self.network = nn.SequentialCell(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = ops.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Dense(embed_dims[-1], num_classes) if num_classes > 0 else ops.Identity()

    def cls_init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else ops.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
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
        cls_out = self.head(ops.Squeeze()((ops.AdaptiveAvgPool2D(output_size=1)(x))))
        return cls_out


def GroupNorm(dim):
    return nn.GroupNorm(1, dim)


def WaveMLP_T_dw(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                    mlp_ratios=mlp_ratios, mode='depthwise', **kwargs)
    model.default_cfg = default_cfgs['wave_T']
    return model


def WaveMLP_T(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                    mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['wave_T']
    return model


def WaveMLP_S(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                    mlp_ratios=mlp_ratios, norm_layer=GroupNorm, **kwargs)
    model.default_cfg = default_cfgs['wave_S']
    return model


def WaveMLP_M(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 18, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                    mlp_ratios=mlp_ratios, norm_layer=GroupNorm, ds_use_norm=False, **kwargs)
    model.default_cfg = default_cfgs['wave_M']
    return model


def WaveMLP_B(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 18, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                    mlp_ratios=mlp_ratios, norm_layer=GroupNorm, ds_use_norm=False, **kwargs)
    model.default_cfg = default_cfgs['wave_B']
    return model
