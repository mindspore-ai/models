# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class DWConv(nn.Cell):
    def __init__(self, dim=32):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, pad_mode='same', has_bias=True, group=dim)

    def construct(self, x, h, w):
        b, _, c = x.shape
        x = x.transpose(0, 2, 1).view(b, c, h, w)
        x = self.dwconv(x)
        x = x.reshape(b, c, -1)
        x = x.transpose(0, 2, 1)
        return x


class Mlp(nn.Cell):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = drop
        if self.drop > 0.:
            self.drop = nn.Dropout(drop)

    def construct(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        if self.drop > 0.:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop > 0.:
            x = self.drop(x)
        return x


class Attention(nn.Cell):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.kv = nn.Dense(dim, dim * 2, has_bias=qkv_bias)
        self.attn_drop = attn_drop
        if self.attn_drop > 0.:
            self.attn_drop = nn.Dropout(attn_drop)
        self.conv = nn.Dense(dim, dim)
        if self.attn_drop > 0.:
            self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(-1)
        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, has_bias=True)
            self.norm = nn.LayerNorm((dim,), epsilon=1e-5)

    def construct(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).transpose(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).transpose(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).transpose(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).transpose(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = ops.BatchMatMul(transpose_b=True)(q, k)
        attn = attn * self.scale
        attn = self.softmax(attn)
        if self.attn_drop > 0.:
            attn = self.attn_drop(attn)
        x = ops.BatchMatMul()(attn, v)
        x = x.transpose(0, 2, 1, 3).reshape(b, n, c)
        x = self.conv(x)
        if self.attn_drop > 0.:
            x = self.proj_drop(x)
        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, sr_ratio=1):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm((dim,), epsilon=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm((dim,), epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, h, w):
        y = self.norm1(x)
        y = self.attn(y, h, w)
        y = x + self.drop_path(y)
        z = self.norm2(y)
        z = self.mlp(z, h, w)
        out = y + self.drop_path(z)
        return out


class OverlapPatchEmbed(nn.Cell):

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=32):
        super(OverlapPatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, pad_mode='pad',
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[0] // 2, patch_size[1] // 2),
                              weight_init='normal', has_bias=True)
        self.norm = nn.LayerNorm((embed_dim,), epsilon=1e-5, gamma_init='ones', beta_init='zeros')

    def construct(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)
        x = x.transpose(0, 2, 1)
        x = self.norm(x)
        return x, h, w


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0)  # always be 0
        self.rand = ops.UniformReal(seed=seed)  # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = ops.Shape()
        self.floor = ops.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)  # b n c
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


class MixVisionTransformer(nn.Cell):
    def __init__(self, in_channel=3, embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8), mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1)):
        super(MixVisionTransformer, self).__init__()
        self.embed_dims = embed_dims
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_channel,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        dpr = [x.item() for x in ops.linspace(Tensor(0, mindspore.float32), Tensor(drop_path_rate, mindspore.float32),
                                              sum(depths)).asnumpy()]
        cur = 0
        self.block1 = nn.CellList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm((embed_dims[0],), epsilon=1e-6)

        cur += depths[0]
        self.block2 = nn.CellList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm((embed_dims[1],), epsilon=1e-6)

        cur += depths[1]
        self.block3 = nn.CellList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm((embed_dims[2],), epsilon=1e-6)

        cur += depths[2]
        self.block4 = nn.CellList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm((embed_dims[3],), epsilon=1e-6)

    def construct(self, x):
        b = x.shape[0]

        outs = []
        x, h, w = self.patch_embed1(x)
        for _, blk in enumerate(self.block1):
            x = blk(x, h, w)
        x = self.norm1(x)
        x = x.reshape(b, h, w, -1).transpose(0, 3, 1, 2)
        outs.append(x)

        x, h, w = self.patch_embed2(x)
        for _, blk in enumerate(self.block2):
            x = blk(x, h, w)
        x = self.norm2(x)
        x = x.reshape(b, h, w, -1).transpose(0, 3, 1, 2)
        outs.append(x)

        x, h, w = self.patch_embed3(x)
        for _, blk in enumerate(self.block3):
            x = blk(x, h, w)
        x = self.norm3(x)
        x = x.reshape(b, h, w, -1).transpose(0, 3, 1, 2)
        outs.append(x)

        x, h, w = self.patch_embed4(x)
        for _, blk in enumerate(self.block4):
            x = blk(x, h, w)
        x = self.norm4(x)
        x = x.reshape(b, h, w, -1).transpose(0, 3, 1, 2)
        outs.append(x)

        return outs


class MitB0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(MitB0, self).__init__(in_channel=3, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                                    attn_drop_rate=0., drop_path_rate=0.1, depths=[2, 2, 2, 2],
                                    sr_ratios=[8, 4, 2, 1])


class MitB1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(MitB1, self).__init__(in_channel=3, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                                    attn_drop_rate=0., drop_path_rate=0.1, depths=[2, 2, 2, 2],
                                    sr_ratios=[8, 4, 2, 1])


class MitB2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(MitB2, self).__init__(in_channel=3, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                                    attn_drop_rate=0., drop_path_rate=0.1, depths=[3, 4, 6, 3],
                                    sr_ratios=[8, 4, 2, 1])


class MitB3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(MitB3, self).__init__(in_channel=3, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                                    attn_drop_rate=0., drop_path_rate=0.1, depths=[3, 4, 18, 3],
                                    sr_ratios=[8, 4, 2, 1])


class MitB4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(MitB4, self).__init__(in_channel=3, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                                    attn_drop_rate=0., drop_path_rate=0.1, depths=[3, 8, 27, 3],
                                    sr_ratios=[8, 4, 2, 1])


class MitB5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(MitB5, self).__init__(in_channel=3, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                                    attn_drop_rate=0., drop_path_rate=0.1, depths=[3, 6, 40, 3],
                                    sr_ratios=[8, 4, 2, 1])
