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

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0)                  # always be 0
        # seed must be 0, if set to other value, it's not rand for multiple call
        self.rand = P.UniformReal(seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
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
        self.drop = nn.Dropout(1. - drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, has_bias=True)
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, 1, 1, has_bias=True)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HireMLP(nn.Cell):
    def __init__(self, dim, attn_drop=0., proj_drop=0., pixel=2, step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        self.pixel = pixel
        self.step = step
        self.step_pad_mode = step_pad_mode
        self.pixel_pad_mode = pixel_pad_mode
        print('pixel: {} pad mode: {} step: {} pad mode: {}'.format(
            pixel, pixel_pad_mode, step, step_pad_mode))

        self.mlp_h1 = nn.Conv2d(dim*pixel, dim//2, 1, has_bias=False)
        self.mlp_h1_norm = nn.BatchNorm2d(dim//2)
        self.mlp_h2 = nn.Conv2d(dim//2, dim*pixel, 1, has_bias=True)
        self.mlp_w1 = nn.Conv2d(dim*pixel, dim//2, 1, has_bias=False)
        self.mlp_w1_norm = nn.BatchNorm2d(dim//2)
        self.mlp_w2 = nn.Conv2d(dim//2, dim*pixel, 1, has_bias=True)
        self.mlp_c = nn.Conv2d(dim, dim, 1, has_bias=True)

        self.act = nn.ReLU()

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1, has_bias=True)
        self.proj_drop = nn.Dropout(1. - proj_drop)

    def construct(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        """

        B, C, H, W = x.shape

        pad_h, pad_w = (
            self.pixel - H % self.pixel) % self.pixel, (self.pixel - W % self.pixel) % self.pixel
        h, w = x.copy(), x.copy()

        if self.step:
            if self.step_pad_mode == 'c':
                if self.step > 0:
                    h_slice = ops.Slice()(h, (0, 0, 0, 0), (B, C, self.step, W))
                    h = ops.Concat(axis=2)((h, h_slice))
                    h = ops.Slice()(h, (0, 0, self.step, 0), (B, C, H, W))
                    w_slice = ops.Slice()(w, (0, 0, 0, 0), (B, C, H, self.step))
                    w = ops.Concat(axis=3)((w, w_slice))
                    w = ops.Slice()(w, (0, 0, 0, self.step), (B, C, H, W))
            else:
                raise NotImplementedError("Invalid pad mode.")

        if self.pixel_pad_mode == '0':
            h = nn.Pad(paddings=((0, 0), (0, 0), (0, pad_h), (0, 0)), mode='CONSTANT')(h)
            w = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, pad_w)), mode='CONSTANT')(w)
        elif self.pixel_pad_mode == 'c':
            if pad_h > 0:
                h_slice = ops.Slice()(h, (0, 0, 0, 0), (B, C, pad_h, W))
                h = ops.Concat(axis=2)((h, h_slice))
            if pad_w > 0:
                w_slice = ops.Slice()(w, (0, 0, 0, 0), (B, C, H, pad_w))
                w = ops.Concat(axis=3)((w, w_slice))
        else:
            raise NotImplementedError("Invalid pad mode.")

        h = (ops.Transpose()(h.reshape(B, C, (H + pad_h) // self.pixel, self.pixel, W), (0, 1, 3, 2, 4))).reshape(
            B, C*self.pixel, (H + pad_h) // self.pixel, W)
        w = (ops.Transpose()(w.reshape(B, C, H, (W + pad_w) // self.pixel, self.pixel), (0, 1, 4, 2, 3))).reshape(
            B, C*self.pixel, H, (W + pad_w) // self.pixel)

        h = self.mlp_h1(h)
        h = self.mlp_h1_norm(h)
        h = self.act(h)
        h = self.mlp_h2(h)

        w = self.mlp_w1(w)
        w = self.mlp_w1_norm(w)
        w = self.act(w)
        w = self.mlp_w2(w)

        h = (ops.Transpose()(h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W), (0, 1, 3, 2, 4))).reshape(
            B, C, H + pad_h, W)
        w = (ops.Transpose()(w.reshape(B, C, self.pixel, H, (W + pad_w) // self.pixel), (0, 1, 3, 4, 2))).reshape(
            B, C, H, W + pad_w)

        h = ops.Slice()(h, (0, 0, 0, 0), (B, C, H, W))
        w = ops.Slice()(w, (0, 0, 0, 0), (B, C, H, W))

        if self.step and self.step_pad_mode == 'c':
            _, _, H_, W_ = h.shape
            h_slice = ops.Slice()(h, (0, 0, H_-self.step, 0), (B, C, self.step, W))
            h = ops.Concat(axis=2)((h_slice, h))
            h = ops.Slice()(h, (0, 0, 0, 0), (B, C, H, W))
            w_slice = ops.Slice()(w, (0, 0, 0, W_-self.step), (B, C, H, self.step))
            w = ops.Concat(axis=3)((w_slice, w))
            w = ops.Slice()(w, (0, 0, 0, 0), (B, C, H, W))

        c = self.mlp_c(x)

        a = ops.AdaptiveAvgPool2D(output_size=(1, 1))(h + w + c)
        a = ops.ExpandDims()(ops.ExpandDims()(
            ops.Softmax(axis=0)(ops.Transpose()(self.reweight(a).reshape(B, C, 3), (2, 0, 1))), -1), -1)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class HireBlock(nn.Cell):

    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 pixel=2, step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = HireMLP(dim, attn_drop=attn_drop, pixel=pixel, step=step,
                            step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else ops.Identity()

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlapping(nn.Cell):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 groups=1, use_norm=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(padding, padding, padding, padding), group=groups, pad_mode='pad', has_bias=True)
        self.norm = norm_layer(embed_dim) if use_norm else ops.Identity()
        self.act = nn.ReLU()

    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, norm_layer=nn.BatchNorm2d, use_norm=True):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(
            3, 3), stride=(2, 2), padding=1, pad_mode='pad', has_bias=True)
        self.norm = norm_layer(
            out_embed_dim) if use_norm else ops.Identity()
        self.act = nn.ReLU()

    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=4., attn_drop=0., drop_path_rate=0., pixel=2, step_stride=2,
                 step_dilation=1, step_pad_mode='c', pixel_pad_mode='c', **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * \
            (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(HireBlock(
            dim, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop_path=block_dpr, pixel=pixel,
            step=(block_idx % step_stride) * step_dilation, step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode))
    blocks = nn.SequentialCell(*blocks)
    return blocks


class HireMLPNet(nn.Cell):
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dims=None, mlp_ratios=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 pixel=None, step_stride=None, step_dilation=None,
                 step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        self.print = ops.Print()

        self.num_classes = num_classes

        self.patch_embed = PatchEmbedOverlapping(
            patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i],
                attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, pixel=pixel[i],
                step_stride=step_stride[i], step_dilation=step_dilation[i],
                step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            network.append(Downsample(embed_dims[i], embed_dims[i+1], 2))

        self.network = nn.SequentialCell(network)

        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Dense(
            embed_dims[-1], num_classes) if num_classes > 0 else ops.Identity()

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
        self.head = nn.Dense(
            self.embed_dim, num_classes) if num_classes > 0 else ops.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        for _, block in enumerate(self.network):
            x = block(x)
        return x

    def construct(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        cls_out = self.head(ops.Squeeze()(
            (ops.AdaptiveAvgPool2D(output_size=1)(x))))
        return cls_out


def hire_mlp_tiny(pretrained=False, **kwargs):
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    pixel = [4, 3, 3, 2]
    step_stride = [2, 2, 3, 2]
    step_dilation = [2, 2, 1, 1]
    step_pad_mode = 'c'
    pixel_pad_mode = 'c'
    model = HireMLPNet(
        layers, embed_dims=embed_dims, patch_size=7, mlp_ratios=mlp_ratios, pixel=pixel,
        step_stride=step_stride, step_dilation=step_dilation,
        step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode, **kwargs)
    model.default_cfg = _cfg()
    return model


def hire_mlp_small(pretrained=False, **kwargs):
    layers = [3, 4, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    pixel = [4, 3, 3, 2]
    step_stride = [2, 2, 3, 2]
    step_dilation = [2, 2, 1, 1]
    step_pad_mode = 'c'
    pixel_pad_mode = 'c'
    model = HireMLPNet(
        layers, embed_dims=embed_dims, patch_size=7, mlp_ratios=mlp_ratios, pixel=pixel,
        step_stride=step_stride, step_dilation=step_dilation,
        step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode, **kwargs)
    model.default_cfg = _cfg()
    return model


def hire_mlp_base(pretrained=False, **kwargs):
    layers = [4, 6, 24, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    pixel = [4, 3, 3, 2]
    step_stride = [2, 2, 3, 2]
    step_dilation = [2, 2, 1, 1]
    step_pad_mode = 'c'
    pixel_pad_mode = 'c'
    model = HireMLPNet(
        layers, embed_dims=embed_dims, patch_size=7, mlp_ratios=mlp_ratios, pixel=pixel,
        step_stride=step_stride, step_dilation=step_dilation,
        step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode, **kwargs)
    model.default_cfg = _cfg()
    return model


def hire_mlp_large(pretrained=False, **kwargs):
    layers = [4, 6, 24, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    pixel = [4, 3, 3, 2]
    step_stride = [2, 2, 3, 2]
    step_dilation = [2, 2, 1, 1]
    step_pad_mode = 'c'
    pixel_pad_mode = 'c'
    model = HireMLPNet(
        layers, embed_dims=embed_dims, patch_size=7, mlp_ratios=mlp_ratios, pixel=pixel,
        step_stride=step_stride, step_dilation=step_dilation,
        step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode, **kwargs)
    model.default_cfg = _cfg()
    return model
