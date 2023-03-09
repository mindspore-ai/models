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
"""efficientnetv2 model define"""
import os

import numpy as np
from mindspore import Tensor, dtype, ops
from mindspore import nn
from mindspore.common import initializer as weight_init

from src.models.var_init import RandomNormal, RandomUniform

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks). """

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(p=drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=dtype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath2D(DropPath):
    """DropPath2D"""

    def __init__(self, drop_prob):
        super(DropPath2D, self).__init__(drop_prob=drop_prob, ndim=2)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    <https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py>
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v


class SiLU(nn.Cell):
    """SiLU"""

    def __init__(self):
        super(SiLU, self).__init__()
        self.ops_sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.ops_sigmoid(x)

    def __repr__(self):
        return "SiLU<x * Sigmoid(x)>"


class SELayer(nn.Cell):
    """SELayer"""

    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc = nn.SequentialCell([
            nn.Conv2d(in_channels=oup, out_channels=inp // reduction,
                      kernel_size=1, has_bias=True),
            SiLU(),
            nn.Conv2d(in_channels=inp // reduction, out_channels=oup,
                      kernel_size=1, has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x):
        y = self.avg_pool(x, [2, 3])
        y = self.fc(y)
        return y * x


def conv_3x3_bn(inp, oup, stride, norm_type):
    """conv_3x3_bn"""
    return nn.SequentialCell([
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, pad_mode='same', has_bias=False),
        norm_type(num_features=oup, momentum=0.9, eps=1e-3),
        SiLU()
    ])


def conv_1x1_bn(inp, oup, norm_type):
    """conv_1x1_bn"""
    return nn.SequentialCell([
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, has_bias=False),
        norm_type(num_features=oup, momentum=0.9, eps=1e-3),
        SiLU()
    ])


class MBConv(nn.Cell):
    """MBConv"""

    def __init__(self, inp, oup, stride, expand_ratio, use_se, norm_type, drop_path_rate=0.):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if self.identity:
            self.drop_path = DropPath2D(drop_path_rate)
        if use_se:
            self.conv = nn.SequentialCell([
                # pw
                nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=1, stride=1, pad_mode='pad',
                          padding=0, has_bias=False),
                norm_type(num_features=hidden_dim, momentum=0.9, eps=1e-3),
                SiLU(),
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride,
                          pad_mode='same', group=hidden_dim, has_bias=False),
                norm_type(num_features=hidden_dim, momentum=0.9, eps=1e-3),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=oup, kernel_size=1, stride=1, has_bias=False),
                norm_type(num_features=oup, momentum=0.9, eps=1e-3),
            ])
        else:
            # fused branch
            if expand_ratio == 1:
                # when expand_ratio == 1: apply dw
                self.conv = nn.SequentialCell([
                    # pw-linear
                    nn.Conv2d(in_channels=hidden_dim, out_channels=oup, kernel_size=3, stride=1, pad_mode='same',
                              has_bias=False),
                    norm_type(num_features=oup, momentum=0.9, eps=1e-3),
                    SiLU(),
                ])
            else:
                self.conv = nn.SequentialCell([
                    # fused
                    nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=3, stride=stride, pad_mode='same',
                              has_bias=False),
                    norm_type(num_features=hidden_dim, momentum=0.9, eps=1e-3),
                    SiLU(),
                    # pw-linear
                    nn.Conv2d(in_channels=hidden_dim, out_channels=oup, kernel_size=1, stride=1, pad_mode='pad',
                              padding=0, has_bias=False),
                    norm_type(num_features=oup, momentum=0.9, eps=1e-3),
                ])

    def construct(self, x):
        if self.identity:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)


class EffNetV2(nn.Cell):
    """EffNetV2"""

    def __init__(self, cfgs, args, num_classes=1000, width_mult=1., drop_out_rate=0., drop_path_rate=0.):
        super(EffNetV2, self).__init__()
        if args.device_target == "Ascend" and int(os.getenv("DEVICE_NUM", args.device_num)) > 1:
            norm_type = nn.SyncBatchNorm
        else:
            norm_type = nn.BatchNorm2d
        self.cfgs = cfgs
        # building first layer
        input_channel = _make_divisible(cfgs[0][1] * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, norm_type=norm_type)]
        # building inverted residual blocks
        block = MBConv
        layers_num = 0
        for _, _, n, _, _ in self.cfgs:
            layers_num += n
        drop_path_rates = np.linspace(0, drop_path_rate, int(layers_num) + 1)
        index = 0
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(
                    block(input_channel, output_channel, s if i == 0 else 1, t,
                          use_se, norm_type, drop_path_rates[index]))
                input_channel = output_channel
                index += 1
        self.features = nn.CellList(layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, norm_type=norm_type)
        self.avgpool = ops.ReduceMean(keep_dims=False)
        self.dropout = nn.Dropout(p=drop_out_rate)
        self.classifier = nn.Dense(in_channels=output_channel, out_channels=num_classes)
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                init_range = 1.0 / np.sqrt(cell.weight.shape[0])
                cell.weight.set_data(weight_init.initializer(RandomUniform(init_range),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            if isinstance(cell, nn.Conv2d):
                out_channel, _, kernel_size_h, kernel_size_w = cell.weight.shape
                stddev = np.sqrt(2 / int(out_channel * kernel_size_h * kernel_size_w))
                cell.weight.set_data(weight_init.initializer(RandomNormal(std=stddev),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def construct(self, x):
        for feature in self.features:
            x = feature(x)
        x = self.conv(x)
        x = self.avgpool(x, [2, 3])
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def effnetv2_s(args):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, args=args, num_classes=args.num_classes, drop_out_rate=args.drop_out_rate,
                    drop_path_rate=args.drop_path_rate)


def effnetv2_m(args):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, args=args, num_classes=args.num_classes, drop_out_rate=args.drop_out_rate,
                    drop_path_rate=args.drop_path_rate)


def effnetv2_l(args):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, args=args, num_classes=args.num_classes, drop_out_rate=args.drop_out_rate,
                    drop_path_rate=args.drop_path_rate)


def effnetv2_xl(args):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, args=args, num_classes=args.num_classes, drop_out_rate=args.drop_out_rate,
                    drop_path_rate=args.drop_path_rate)
