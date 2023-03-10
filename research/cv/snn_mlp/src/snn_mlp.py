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
from itertools import repeat
import collections.abc
import numpy
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))

def MyNorm(dim):
    return nn.GroupNorm(1, dim)



class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0) # always be 0
        self.rand = P.UniformReal(seed=seed) # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x) # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(p=drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, has_bias=True)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, has_bias=True)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LIFModule(nn.Cell):
    def __init__(self, dim, lif_bias=True, proj_drop=0.,
                 lif=-1, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=-1.):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, padding=0, group=1, has_bias=lif_bias)
        self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, padding=0, group=1, has_bias=lif_bias)
        self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, padding=0, group=1, has_bias=lif_bias)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, padding=0, group=1, has_bias=lif_bias)

        self.actn = nn.GELU()

        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)
        self.norm3 = MyNorm(dim)

        self.lif1 = LIFSpike(lif=lif, fix_tau=lif_fix_tau, fix_vth=lif_fix_vth,
                             init_tau=lif_init_tau, init_vth=lif_init_vth, dim=2)
        self.lif2 = LIFSpike(lif=lif, fix_tau=lif_fix_tau, fix_vth=lif_fix_vth,
                             init_tau=lif_init_tau, init_vth=lif_init_vth, dim=3)
        self.dw1 = nn.Conv2d(dim, dim, 3, 1, group=dim, has_bias=lif_bias)


    def construct(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actn(x)
        x = self.dw1(x)
        x = self.norm2(x)
        x = self.actn(x)
        x_lif_lr = self.lif1(x)
        x_lif_td = self.lif2(x)

        x_lr = self.conv2_1(x_lif_lr)
        x_td = self.conv2_2(x_lif_td)

        x_lr = self.actn(x_lr)
        x_td = self.actn(x_td)

        x = x_lr + x_td
        x = self.norm3(x)
        x = self.conv3(x)

        return x


class PatchMerging(nn.Cell):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, 1, has_bias=False)
        self.norm = norm_layer(4 * dim)

    def construct(self, x):
        x0 = x[:, :, 0::2, 0::2]  # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]  # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]  # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]  # B H/2 W/2 C
        x = ops.Concat(1)([x0, x1, x2, x3])  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class LIFBlock(nn.Cell):
    def __init__(self, dim,
                 mlp_ratio=4., lif_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 lif=-1, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=-1.):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.lif_module = LIFModule(dim, lif_bias=lif_bias, proj_drop=drop,
                                    lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                                    lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else ops.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        shortcut = x
        x = self.norm1(x)

        # lif block
        x = self.lif_module(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Cell):
    def __init__(self, dim, depth,
                 mlp_ratio=4., lif_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 lif=-1, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=-1.):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.CellList([
            LIFBlock(dim=dim,
                     mlp_ratio=mlp_ratio,
                     lif_bias=lif_bias,
                     drop=drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer,
                     lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                     lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def construct(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Cell):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def construct(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SNNMLP(nn.Cell):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=None,
                 mlp_ratio=4., lif_bias=True,
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=MyNorm, patch_norm=True,
                 lif=4, lif_fix_tau=False, lif_fix_vth=False,
                 lif_init_tau=0.25, lif_init_vth=0.25, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x for x in numpy.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               lif_bias=lif_bias,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                               lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else ops.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = ops.AdaptiveAvgPool2D(output_size=(1, 1))(x)
        x = ops.Flatten()(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n, tau, Vth):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = ops.Greater()(u_t1_n1, Vth).astype(W_mul_o_t1_n.dtype)
    r_t1_n1 = ops.ReLU()(u_t1_n1-Vth)+Vth
    return u_t1_n1, o_t1_n1, r_t1_n1


class LIFSpike(nn.Cell):
    def __init__(self, lif, fix_tau=False, fix_vth=False, init_tau=0.25, init_vth=-1., dim=2):
        super(LIFSpike, self).__init__()
        if fix_tau:
            self.tau = init_tau
        else:
            self.tau = mindspore.Parameter(mindspore.Tensor([init_tau]))
        if fix_vth:
            self.Vth = init_vth
        else:
            self.Vth = mindspore.Parameter(mindspore.Tensor([init_vth]))
        self.lif = lif
        self.dim = dim

    def construct(self, x):
        if self.lif == 0:
            nums = 1
            endnum = x.shape[self.dim]
            lif_step = x.shape[self.dim]
        else:
            nums = x.shape[self.dim] // self.lif
            endnum = x.shape[self.dim] - x.shape[self.dim] % self.lif
            lif_step = self.lif
        if self.dim == 2:
            u = ops.Zeros()((x.shape[0], x.shape[1], nums, x.shape[3]), mindspore.float32)
            o = ops.Zeros()((x.shape[0], x.shape[1], nums, x.shape[3]), mindspore.float32)
            for step in range(lif_step):
                u, o, x[:, :, step:endnum:lif_step, :] = \
                    state_update(u, o, x[:, :, step:endnum:lif_step, :], self.tau, self.Vth)
        elif self.dim == 3:
            u = ops.Zeros()((x.shape[0], x.shape[1], x.shape[2], nums), mindspore.float32)
            o = ops.Zeros()((x.shape[0], x.shape[1], x.shape[2], nums), mindspore.float32)
            for step in range(lif_step):
                u, o, x[:, :, :, step:endnum:lif_step] = \
                    state_update(u, o, x[:, :, :, step:endnum:lif_step], self.tau, self.Vth)
        return x



def snnmlp_t(drop_path_rate=0.2, drop_rate=0.0, **kwargs):
    return SNNMLP(
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        mlp_ratio=4,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        patch_norm=True)


def snnmlp_s(drop_path_rate=0.3, drop_rate=0.0, **kwargs):
    return SNNMLP(
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        mlp_ratio=4,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        patch_norm=True)


def snnmlp_b(drop_path_rate=0.5, drop_rate=0.0, **kwargs):
    return SNNMLP(
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        mlp_ratio=4,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        patch_norm=True)
