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
import math
from src.model_utils.misc import DropPath1D, to_2tuple
import mindspore
from mindspore import nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.common.initializer import initializer, TruncatedNormal

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_ti_patch16': _cfg(
        input_size=(3, 192, 192), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_s_patch16': _cfg(
        input_size=(3, 256, 256), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_m_patch16': _cfg(
        input_size=(3, 256, 256), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16': _cfg(
        input_size=(3, 256, 256), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features,
                            out_channels=hidden_features)
        self.act = act_layer(approximate=False)
        self.fc2 = nn.Dense(in_channels=hidden_features,
                            out_channels=out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# cross-fusion of outer to Inner
class ResBlock_inner(nn.Cell):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim

        self.conv1x1 = nn.Conv2d(in_channels=dim * 16, out_channels=dim, kernel_size=1, pad_mode='pad', padding=0,
                                 has_bias=True, weight_init='Zero', bias_init='Zero')

        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, pad_mode='pad',
                               padding=(kernel_size - 1) // 2, has_bias=True, weight_init='Zero', bias_init='Zero')
        self.norm = nn.LayerNorm((dim,), epsilon=1e-05)
        self.act = nn.GELU(approximate=False)

    def construct(self, outer, inner, H, W):
        outer = ops.Reshape()(ops.Transpose()(outer, (0, 2, 1,)),
                              (outer.shape[0], -1, H, W,))

        inner = ops.Transpose()(ops.Reshape()(
            ops.Transpose()(ops.Reshape()(
                inner, (outer.shape[0], H, W, 4, 4, inner.shape[-1],)), (0, 1, 3, 2, 4, 5,)),
            (outer.shape[0], H * 4, W * 4, inner.shape[-1],)), (0, 3, 1, 2,))
        inner = inner + self.act(self.norm(self.conv2(
            inner + ops.interpolate(self.conv1x1(outer), scales=(1.0, 1.0, 4.0, 4.0),
                                    coordinate_transformation_mode='half_pixel', mode="bilinear")).reshape(
                                        outer.shape[0], -1, 4 * H * 4 * W).transpose(0, 2, 1)).reshape(
                                            outer.shape[0], 4 * H, 4 * W, -1).transpose(0, 3, 1, 2))

        inner = ops.Reshape()(
            ops.Transpose()(ops.Reshape()(
                inner, (outer.shape[0], inner.shape[1], H, 4, W, 4,)), (0, 2, 4, 3, 5, 1,)),
            (outer.shape[0] * H * W, 4 * 4, -1,))

        return inner


# cross-fusion of inner to outer
class ResBlock_outer(nn.Cell):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim

        self.conv4x4 = nn.Conv2d(in_channels=dim // 16, out_channels=dim, kernel_size=4, stride=4, pad_mode='same',
                                 padding=0, has_bias=True, weight_init='Zero', bias_init='Zero')

        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, pad_mode='pad',
                               padding=(kernel_size - 1) // 2, has_bias=True, weight_init='Zero', bias_init='Zero')
        self.norm = nn.LayerNorm((dim,), epsilon=1e-05)
        self.act = nn.GELU(approximate=False)

    def construct(self, outer, inner, H, W):
        outer = ops.Reshape()(ops.Transpose()(outer, (0, 2, 1,)),
                              (outer.shape[0], -1, H, W,))

        inner = ops.Transpose()(ops.Reshape()(
            ops.Transpose()(ops.Reshape()(
                inner, (outer.shape[0], H, W, 4, 4, inner.shape[-1],)), (0, 1, 3, 2, 4, 5,)),
            (
                outer.shape[0], H * 4,
                W * 4, inner.shape[-1],)), (0, 3, 1, 2,))
        outer = outer + self.act(self.norm(
            self.conv2(self.conv4x4(inner) + outer).reshape(
                outer.shape[0], -1, H * W).transpose((0, 2, 1))).reshape(
                    outer.shape[0], H, W, -1).transpose(0, 3, 1, 2))

        outer = ops.Reshape()(
            outer, (outer.shape[0], -1, H * W,)).transpose(0, 2, 1)

        return outer


class SE(nn.Cell):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.SequentialCell([
            nn.LayerNorm((dim,), epsilon=1e-05),
            nn.Dense(in_channels=dim, out_channels=hidden_dim),
            nn.ReLU(),
            nn.Dense(in_channels=hidden_dim, out_channels=dim),
            nn.Tanh()
        ])

    def construct(self, x):
        a = ops.ReduceMean()(True, x, 1)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


# Add det_dokens to Attention
class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.kv = nn.Dense(
            in_channels=dim, out_channels=dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.pool = nn.AvgPool2d(
                kernel_size=sr_ratio, stride=sr_ratio, pad_mode='valid')
            self.linear = nn.Dense(in_channels=dim, out_channels=dim)
            self.norm = nn.LayerNorm((dim,), epsilon=1e-05)

        self.softmax = ops.Softmax(-1)

    def construct(self, x, H, W, relative_pos=None, det_token=None, cross=False, mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).transpose(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = ops.Reshape()(ops.Transpose()(x, (0, 2, 1,)), (B, C, H, W,))

            x_ = self.pool(x_).reshape(B, C, -1).transpose(0, 2, 1)

            x_ = self.norm(self.linear(x_))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C //
                                     self.num_heads).transpose(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C //
                                    self.num_heads).transpose(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (ops.matmul(q, k.transpose(0, 1, 3, 2))) * self.scale
        if relative_pos is not None:
            relative_pos = ops.interpolate(relative_pos, sizes=(attn.shape[2], attn.shape[3]), mode='bicubic',
                                           coordinate_transformation_mode='half_pixel')
            attn += relative_pos
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (ops.matmul(attn, v)).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if det_token is not None:

            if cross:

                B_det, N_det, C = det_token.shape

                q_det = self.q(det_token).reshape(B_det, N_det, self.num_heads,
                                                  C // self.num_heads).transpose(0, 2, 1, 3)
                kv_det = self.kv(det_token).reshape(B_det, -1, 2, self.num_heads,
                                                    C // self.num_heads).transpose(2, 0, 3, 1, 4)
                k_det, v_det = kv_det[0], kv_det[1]
                k_det = ops.Concat(2)([k, k_det])
                v_det = ops.Concat(2)([v, v_det])
                attn_det = (ops.matmul(
                    q_det, k_det.transpose(0, 1, 3, 2))) * self.scale
                attn_det = self.softmax(attn_det)
                attn_det = self.attn_drop(attn_det)

                det_token = ops.matmul(attn_det, v_det).transpose(
                    0, 2, 1, 3).reshape(B_det, N_det, C)
                det_token = self.proj(det_token)
                det_token = self.proj_drop(det_token)

            else:

                B_det, N_det, C = det_token.shape
                q_det = self.q(det_token).reshape(B_det, N_det, self.num_heads, C // self.num_heads).transpose(0, 2, 1,
                                                                                                               3)
                kv_det = self.kv(det_token).reshape(B_det, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0,
                                                                                                                 3, 1,
                                                                                                                 4)
                k_det, v_det = kv_det[0], kv_det[1]
                attn_det = (ops.matmul(
                    q_det, k_det.transpose(0, 1, 3, 2))) * self.scale
                attn_det = self.softmax(attn_det)
                attn_det = self.attn_drop(attn_det)

                det_token = (ops.matmul(attn_det, v_det)).transpose(
                    0, 2, 1, 3).reshape(B_det, N_det, C)
                det_token = self.proj(det_token)
                det_token = self.proj_drop(det_token)

            return x, det_token

        return x


class InnercrossAttention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.kv = nn.Dense(
            in_channels=dim, out_channels=dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = ops.Softmax(-1)

    def construct(self, x_det, x_inner):
        B, N, C = x_det.shape
        q = self.q(x_det).reshape(B, N, self.num_heads, C //
                                  self.num_heads).transpose(0, 2, 1, 3)

        kv = self.kv(x_inner).reshape(B, -1, 2, self.num_heads,
                                      C // self.num_heads).transpose(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (ops.matmul(q, k.transpose(-2, -1))) * self.scale

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (ops.matmul(attn, v)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# outer down-sample
class SentenceAggregation(nn.Cell):
    """ Sentence Aggregation
    """

    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm((dim_in,), epsilon=1e-05)
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=2 * stride - 1, stride=stride,
                      pad_mode='pad', padding=stride - 1, has_bias=True),
        ])

        self.ratio = dim_out // dim_in

    def construct(self, x, H, W, det_token):
        B, _, C = x.shape  # B, N, C
        x = self.norm(x)
        x_ = ops.Reshape()(x.transpose(0, 2, 1), (B, C, H, W,))
        x_ = self.conv(x_)
        H, W = math.ceil(H / self.stride), math.ceil(W / self.stride)
        x_ = ops.Reshape()(x_, (B, -1, H * W,)).transpose(0, 2, 1)

        # 注意，这里是det token直接复制，从而让det token与outer的channel保持一致
        det_token = ops.Tile()(det_token, (1, 1, self.ratio,))

        return x_, H, W, det_token


# inner down-sample
class WordAggregation(nn.Cell):
    """ Word Aggregation
    """

    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm((dim_in,), epsilon=1e-05)
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=2 * stride - 1, stride=stride,
                      pad_mode='pad', padding=stride - 1, has_bias=True),
        ])

    def construct(self, x, H_out, W_out, H_in, W_in):
        _, M, C = x.shape  # B*N, M, C
        x = self.norm(x)
        x = ops.Reshape()(x, (-1, H_out, W_out, H_in, W_in, C,))
        pad_input = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad_input:
            pad = ops.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0), (0, H_out % 2), (0, W_out % 2)))
            x = ops.Transpose()(x, (0, 3, 4, 5, 1, 2,))
            x = pad(x)
            x = ops.Transpose()(x, (0, 4, 5, 1, 2, 3,))
        x1 = x[:, 0::2, 0::2, :, :, :]  # B, H/2, W/2, H_in, W_in, C
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]
        # B, H/2, W/2, 2*H_in, 2*W_in, C
        x = ops.Concat(4)([ops.Concat(3)([x1, x2]), ops.Concat(3)([x3, x4])])
        x = ops.Transpose()(ops.Reshape()(x, (-1, 2 * H_in, 2 * W_in, C,)),
                            (0, 3, 1, 2,))  # B_N/4, C, 2*H_in, 2*W_in
        x = self.conv(x)  # B_N/4, C, H_in, W_in
        x = ops.Reshape()(x, (-1, self.dim_out, M,)).transpose(0, 2, 1)
        return x


def unfold(img, kernel_size, stride=1, pad=0, dilation=1):
    """
    unfold function
    """
    batch_num, channel, height, width = img.shape
    out_h = (height + pad + pad - kernel_size -
             (kernel_size - 1) * (dilation - 1)) // stride + 1
    out_w = (width + pad + pad - kernel_size -
             (kernel_size - 1) * (dilation - 1)) // stride + 1

    img = np.pad(img, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((batch_num, channel, kernel_size, kernel_size, out_h, out_w)).astype(img.dtype)

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = np.reshape(col, (batch_num, channel * kernel_size * kernel_size, out_h * out_w))

    return col


class Stem(nn.Cell):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_chans=3, outer_dim=768, inner_dim=24):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.inner_dim = inner_dim
        self.num_patches = img_size[0] // 8 * img_size[1] // 8
        self.num_words = 16
        self.common_conv = nn.SequentialCell([
            nn.Conv2d(in_channels=in_chans, out_channels=inner_dim * 2, kernel_size=3, stride=2, pad_mode='pad',
                      padding=1, has_bias=True),
            nn.BatchNorm2d(inner_dim * 2),
            nn.ReLU(),
        ])
        self.inner_convs = nn.SequentialCell([
            nn.Conv2d(in_channels=inner_dim * 2, out_channels=inner_dim, kernel_size=3, stride=1, pad_mode='pad',
                      padding=1, has_bias=True),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
        ])
        self.outer_convs = nn.SequentialCell([
            nn.Conv2d(in_channels=inner_dim * 2, out_channels=inner_dim * 4, kernel_size=3, stride=2, pad_mode='pad',
                      padding=1, has_bias=True),
            nn.BatchNorm2d(inner_dim * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=inner_dim * 4, out_channels=inner_dim * 8, kernel_size=3, stride=2, pad_mode='pad',
                      padding=1, has_bias=True),
            nn.BatchNorm2d(inner_dim * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=inner_dim * 8, out_channels=outer_dim, kernel_size=3, stride=1, pad_mode='pad',
                      padding=1, has_bias=True),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(),
        ])
        self.unfold = unfold

    def construct(self, x):
        B, _, H, W = x.shape
        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4
        x = self.common_conv(x)
        inner_tokens = self.inner_convs(x)  # B, C, H, W
        inner_tokens = self.unfold(inner_tokens, 4, 4).transpose(0, 2, 1)
        inner_tokens = np.reshape(inner_tokens, (B * H_out * W_out, self.inner_dim, H_in * W_in)).transpose(
            (0, 2, 1))  # B*N, C, 4*4
        outer_tokens = self.outer_convs(x)
        outer_tokens = ops.transpose(outer_tokens, (0, 2, 3, 1))
        outer_tokens = np.reshape(outer_tokens, (B, H_out * W_out, -1))
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)


class Block(nn.Cell):
    """ TNT Block
    """

    def __init__(self, outer_dim, inner_dim, outer_head, inner_head, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1, det_number=100, cross_inner=False, inner_conv=False):
        super().__init__()
        self.has_inner = inner_dim > 0
        self.cross_inner = cross_inner
        self.inner_conv_ = inner_conv
        if self.has_inner:
            self.inner_dim = inner_dim
            self.inner_norm1 = norm_layer(
                [num_words * inner_dim], epsilon=1e-5)
            self.inner_attn = Attention(
                inner_dim, num_heads=inner_head, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer(
                [num_words * inner_dim], epsilon=1e-5)
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer([num_words * inner_dim], epsilon=1e-5)
            self.proj = nn.Dense(
                in_channels=num_words * inner_dim, out_channels=outer_dim, has_bias=False)
            self.proj_norm2 = norm_layer([outer_dim], epsilon=1e-5)

            if inner_conv:
                self.inner_conv = nn.SequentialCell([
                    nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=7, stride=1, pad_mode='pad',
                              padding=3, group=inner_dim, has_bias=False),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=1, stride=1, pad_mode='pad',
                              padding=0, has_bias=False),
                    nn.ReLU(),
                ])

        if self.cross_inner and self.has_inner:
            self.inner_cross_attn = InnercrossAttention(dim=outer_dim, num_heads=outer_head, qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                                        sr_ratio=sr_ratio)
            self.inner_cross_norm = norm_layer([outer_dim], epsilon=1e-5)

        # Outer
        self.outer_norm1 = norm_layer([outer_dim], epsilon=1e-5)

        self.outer_norm_det1 = norm_layer([outer_dim], epsilon=1e-5)

        self.sr_ratio = sr_ratio
        self.outer_attn = Attention(
            outer_dim, num_heads=outer_head, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath1D(
            drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer([outer_dim], epsilon=1e-5)
        self.outer_norm_det2 = norm_layer([outer_dim], epsilon=1e-5)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.det_number = det_number
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)

    def construct(self, x, outer_tokens, det_tokens, H_out, W_out, H_in, W_in, relative_pos, det_pos, cross=False,
                  mask=None):
        B, N, _ = ops.Shape()(outer_tokens)

        det_tokens = det_tokens + self.det_pos_linear(det_pos)

        if self.has_inner:
            x = x + self.drop_path(
                self.inner_attn(
                    self.inner_norm1(ops.Reshape()(x, (B, N, -1,))
                                     ).reshape(B * N, H_in * W_in, -1), H_in,
                    W_in))  # B*N, k*k, c

            x = x + self.drop_path(
                self.inner_mlp(
                    self.inner_norm2(ops.Reshape()(x, (B, N, -1,))).reshape(B * N, H_in * W_in, -1)))  # B*N, k*k, c

            outer_tokens = outer_tokens + self.proj_norm2(
                self.proj(self.proj_norm1(ops.Reshape()(x, (B, N, -1,)))))  # B, N, C

        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(
                self.outer_attn(self.outer_norm1(outer_tokens), H_out, W_out, relative_pos))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + \
                self.drop_path(tmp_ + self.se_layer(tmp_))
        else:

            outer_tokens_, det_tokens_ = self.outer_attn(self.outer_norm1(outer_tokens), H_out, W_out, relative_pos,
                                                         self.outer_norm_det1(det_tokens), cross=cross, mask=mask)

            outer_tokens = outer_tokens + self.drop_path(outer_tokens_)

            det_tokens = det_tokens + self.drop_path(det_tokens_)

            outer_tokens__ = self.outer_mlp(self.outer_norm2(outer_tokens))

            det_tokens__ = self.outer_mlp(self.outer_norm_det2(det_tokens))

            outer_tokens = outer_tokens + self.drop_path(outer_tokens__)

            det_tokens = det_tokens + self.drop_path(det_tokens__)

        return x, outer_tokens, det_tokens


class Stage(nn.Cell):
    """ PyramidTNT stage
    """

    def __init__(self, num_blocks, outer_dim, inner_dim, outer_head, inner_head, num_patches, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [
            drop_path] * num_blocks

        for j in range(num_blocks):
            if j == 0:
                _inner_dim = inner_dim
            elif j == 1 and num_blocks > 6:
                _inner_dim = inner_dim
            else:
                _inner_dim = -1
            blocks.append(Block(
                outer_dim, _inner_dim, outer_head=outer_head, inner_head=inner_head,
                num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[j], act_layer=act_layer, norm_layer=norm_layer,
                se=se, sr_ratio=sr_ratio, cross_inner=False, inner_conv=False))

        self.blocks = nn.CellList(blocks)

        self.relative_pos = None

    def construct(self, inner_tokens, outer_tokens, det_tokens, H_out, W_out, H_in, W_in, det_pos, cross=False,
                  mask=None):
        for blk in self.blocks:
            inner_tokens, outer_tokens, det_tokens = blk(inner_tokens, outer_tokens, det_tokens, H_out, W_out, H_in,
                                                         W_in, self.relative_pos,
                                                         det_pos, cross=cross, mask=mask)
        return inner_tokens, outer_tokens, det_tokens


class PyramidTNT(nn.Cell):
    """ PyramidTNT (Transformer in Transformer) for computer vision
    """

    def __init__(self, configs=None, img_size=224, in_chans=3, num_classes=1000, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.num_classes = num_classes
        depths = configs['depths']
        outer_dims = configs['outer_dims']
        inner_dims = configs['inner_dims']
        outer_heads = configs['outer_heads']
        inner_heads = configs['inner_heads']
        self.patch_size = [8, 16, 32, 64]
        self.outer_dims = outer_dims
        self.inner_dims = inner_dims
        self.img_size = to_2tuple(img_size)
        sr_ratios = [4, 2, 1, 1]
        self.patch_embed = Stem(
            img_size=img_size, in_chans=in_chans, outer_dim=outer_dims[0], inner_dim=inner_dims[0])
        num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        self.outer_pos = mindspore.Parameter(ops.Zeros()(
            (1, num_patches, outer_dims[0]), mindspore.float32))
        self.inner_pos = mindspore.Parameter(ops.Zeros()(
            (1, num_words, inner_dims[0]), mindspore.float32))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed = self.outer_pos
        self.num_channels = outer_dims

        #  for stochastic depth decay rule
        depth = 0
        start = mindspore.Tensor(0, mindspore.float32)
        end = mindspore.Tensor(drop_path_rate, mindspore.float32)
        dpr = [x.item() for x in ops.LinSpace()(
            start, end, sum(depths)).asnumpy()]
        self.word_merges = nn.CellList([])
        self.sentence_merges = nn.CellList([])
        self.stages = nn.CellList([])
        self.word_merges = nn.CellList([])
        self.ResBlocks_inner = nn.CellList([])
        self.ResBlocks_outer = nn.CellList([])
        self.UpsampleConv_outer = nn.CellList([])
        self.UpsampleConv_inner = nn.CellList([])
        for i in range(4):
            if i > 0:
                self.word_merges.append(WordAggregation(
                    inner_dims[i - 1], inner_dims[i], stride=2))
                self.sentence_merges.append(SentenceAggregation(
                    outer_dims[i - 1], outer_dims[i], stride=2))

                self.UpsampleConv_outer.append(
                    nn.Conv2d(in_channels=outer_dims[i], out_channels=outer_dims[i - 1], kernel_size=1, pad_mode='pad',
                              padding=0, has_bias=False, weight_init='Zero'))  # FPN中的channel变换，用的是点卷积
                self.UpsampleConv_inner.append(
                    nn.Conv2d(in_channels=inner_dims[i], out_channels=inner_dims[i - 1], kernel_size=1, pad_mode='pad',
                              padding=0, has_bias=False, weight_init='Zero'))

            self.stages.append(Stage(depths[i], outer_dim=outer_dims[i], inner_dim=inner_dims[i],
                                     outer_head=outer_heads[i], inner_head=inner_heads[i],
                                     num_patches=num_patches // (2 ** i) // (2 ** i), num_words=num_words,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=dpr[depth:depth + depths[i]
                                                   ], norm_layer=norm_layer, se=se,
                                     sr_ratio=sr_ratios[i])
                               )

            self.ResBlocks_inner.append(ResBlock_inner(inner_dims[i]))
            self.ResBlocks_outer.append(ResBlock_outer(outer_dims[i]))
            depth += depths[i]
        self.word_merges.append(WordAggregation(
            inner_dims[-1], inner_dims[-1], stride=2))
        self.sentence_merges.append(SentenceAggregation(
            outer_dims[-1], outer_dims[-1], stride=2))

    def construct(self, x, mask):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        inner_tokens, outer_tokens, (H_out,
                                     W_out), (H_in, W_in) = self.patch_embed(x)
        inner_tokens += self.inner_pos  # B*N, 8*8, C

        if self.patch_pos_embed.shape[1] != inner_tokens.shape[0] // B:
            temp_pos_embed = self.InterpolateInitPosEmbed(
                self.patch_pos_embed, img_size=(H, W))
        else:
            temp_pos_embed = self.patch_pos_embed

        outer_tokens += self.pos_drop(temp_pos_embed)
        det_tokens = ops.broadcast_to(self.det_token, (B, -1, -1))
        patch_outs = []
        inner_outs = []
        cross = False
        for i in range(4):
            if i == 3:
                cross = True
            if i > 0:
                inner_tokens = self.word_merges[i -
                                                1](new_inner, H_out, W_out, H_in, W_in)
                outer_tokens, H_out, W_out, det_tokens = self.sentence_merges[i - 1](new_outer, H_out, W_out,
                                                                                     det_tokens)

            # TNT
            inner_tokens, outer_tokens, det_tokens = self.stages[i](inner_tokens, outer_tokens, det_tokens, H_out,
                                                                    W_out, H_in, W_in,
                                                                    self.det_pos, cross=cross, mask=None)

            # cross after each stage
            new_outer = self.ResBlocks_outer[i](
                outer_tokens, inner_tokens, H_out, W_out)
            new_inner = self.ResBlocks_inner[i](
                outer_tokens, inner_tokens, H_out, W_out)
            if i >= 0:  # keep ourput of stages
                patch_outs.append(ops.Reshape()(ops.Transpose()(
                    new_outer, (0, 2, 1,)), (B, -1, H_out, W_out,)))
                inner_outs.append(
                    ops.Transpose()(ops.Reshape()(
                        ops.Transpose()(ops.Reshape()(new_inner, (B, H_out, W_out, H_in, W_in, self.inner_dims[i],)),
                                        (0, 1, 3, 2, 4, 5,)), (B, H_out * H_in, W_out * W_in, self.inner_dims[i],)),
                                    (0, 3, 1, 2,)))
        patch_outs[2] = patch_outs[2] + ops.interpolate(self.UpsampleConv_outer[2](patch_outs[3]),
                                                        sizes=(
                                                            patch_outs[2].shape[2], patch_outs[2].shape[3]),
                                                        coordinate_transformation_mode='half_pixel', mode='bilinear')

        patch_outs[1] = patch_outs[1] + ops.interpolate(self.UpsampleConv_outer[1](patch_outs[2]),
                                                        sizes=(
                                                            patch_outs[1].shape[2], patch_outs[1].shape[3]),
                                                        coordinate_transformation_mode='half_pixel', mode='bilinear')

        patch_outs[0] = patch_outs[0] + ops.interpolate(self.UpsampleConv_outer[0](patch_outs[1]),
                                                        sizes=(
                                                            patch_outs[0].shape[2], patch_outs[0].shape[3]),
                                                        coordinate_transformation_mode='half_pixel', mode='bilinear')

        inner_outs[2] = inner_outs[2] + ops.interpolate(self.UpsampleConv_inner[2](inner_outs[3]),
                                                        sizes=(
                                                            inner_outs[2].shape[2], inner_outs[2].shape[3]),
                                                        coordinate_transformation_mode='half_pixel', mode='bilinear')

        inner_outs[1] = inner_outs[1] + ops.interpolate(self.UpsampleConv_inner[1](inner_outs[2]),
                                                        sizes=(
                                                            inner_outs[1].shape[2], inner_outs[1].shape[3]),
                                                        coordinate_transformation_mode='half_pixel', mode='bilinear')

        #
        inner_outs[0] = inner_outs[0] + ops.interpolate(self.UpsampleConv_inner[0](inner_outs[1]),
                                                        sizes=(
                                                            inner_outs[0].shape[2], inner_outs[0].shape[3]),
                                                        coordinate_transformation_mode='half_pixel', mode='bilinear')

        det_tgt = ops.Transpose()(det_tokens, (0, 2, 1,))

        det_pos = ops.Transpose()(self.det_pos, (0, 2, 1))

        return patch_outs, det_tgt, det_pos, inner_outs

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):

        patch_pos_embed = pos_embed
        patch_pos_embed = patch_pos_embed.transpose((0, 2, 1))
        B, E, _ = patch_pos_embed.shape

        P_H, P_W = self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[0]
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size[0], W // self.patch_size[0]
        patch_pos_embed = ops.interpolate(patch_pos_embed, sizes=(new_P_H, new_P_W), mode='bilinear',
                                          coordinate_transformation_mode="half_pixel")
        patch_pos_embed = patch_pos_embed.view(B, E, -1).transpose(0, 2, 1)

        return patch_pos_embed

    # noinspection DuplicatedCode
    def finetune_det(self, method, cross_indices=2, img_size=(256, 256), det_token_num=100, pos_dim=256):
        # finetune original pyramid TNT backbone
        self.det_token_num = det_token_num
        det_token = initializer(TruncatedNormal(sigma=.02), [
            1, det_token_num, self.outer_dims[0]], mindspore.float32)
        self.det_token = mindspore.Parameter(det_token)
        det_pos_embed = initializer(TruncatedNormal(sigma=.02), [
            1, det_token_num, pos_dim], mindspore.float32)
        self.det_pos = mindspore.Parameter(det_pos_embed)
        patch_pos_embed = self.pos_embed
        patch_pos_embed = patch_pos_embed.transpose(0, 2, 1)
        B, E, _ = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[0]
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size[0], W // self.patch_size[0]
        self.num_patches = new_P_H * new_P_W
        patch_pos_embed = ops.interpolate(patch_pos_embed, sizes=(new_P_H, new_P_W), mode='bilinear',
                                          coordinate_transformation_mode='half_pixel')
        patch_pos_embed = patch_pos_embed.view(B, E, -1).transpose((0, 2, 1))
        self.patch_pos_embed = mindspore.Parameter(patch_pos_embed.asnumpy())
        self.img_size = img_size

        for k, stage in enumerate(self.stages):
            for block in stage.blocks:
                block.det_pos_linear = nn.Dense(pos_dim, self.outer_dims[k])


def ptnt_ti_patch16(pretrained=False, **kwargs):
    outer_dim = 80
    inner_dim = 5
    outer_head = 2
    inner_head = 1
    configs = {
        'depths': [2, 6, 3, 2],
        'outer_dims': [outer_dim, outer_dim * 2, outer_dim * 4, outer_dim * 4],
        'inner_dims': [inner_dim, inner_dim * 2, inner_dim * 4, inner_dim * 4],
        'outer_heads': [outer_head, outer_head * 2, outer_head * 4, outer_head * 4],
        'inner_heads': [inner_head, inner_head * 2, inner_head * 4, inner_head * 4],
    }

    drop_path_rate = 0.1
    model = PyramidTNT(configs=configs, img_size=192,
                       drop_path_rate=drop_path_rate, qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_ti_patch16']
    return model
