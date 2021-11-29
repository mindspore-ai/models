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
# ===========================================================================
"""Operations used in NAS Base Cell"""
import mindspore.nn as nn
from mindspore import ops

from .bn import NormLeakyReLU, BatchNormalization

OPS = {
    'none': lambda channels, stride, momentum, eps, affine, use_abn, parallel: Zero(stride),
    'avg_pool_3x3': lambda channels, stride, momentum, eps, affine, use_abn, parallel: nn.AvgPool2d(3, stride=stride, pad_mode='same'),
    'max_pool_3x3': lambda channels, stride, momentum, eps, affine, use_abn, parallel: nn.MaxPool2d(3, stride=stride, pad_mode='same'),
    'skip_connect': lambda channels, stride, momentum, eps, affine, use_abn, parallel: Identity() if stride == 1 else FactorizedReduce(channels, channels, affine=affine),
    'sep_conv_3x3': lambda channels, stride, momentum, eps, affine, use_abn, parallel: SepConv(channels, channels, 3, stride, 1, momentum, eps, affine, parallel=parallel),
    'sep_conv_5x5': lambda channels, stride, momentum, eps, affine, use_abn, parallel: SepConv(channels, channels, 5, stride, 2, momentum, eps, affine, parallel=parallel),
    'dil_conv_3x3': lambda channels, stride, momentum, eps, affine, use_abn, parallel: DilConv(channels, channels, 3, stride, 2, 2, momentum, eps, affine, use_abn=use_abn, parallel=parallel),
    'dil_conv_5x5': lambda channels, stride, momentum, eps, affine, use_abn, parallel: DilConv(channels, channels, 5, stride, 4, 2, momentum, eps, affine, use_abn=use_abn, parallel=parallel)
}


class SepConv(nn.Cell):
    """Sepconv"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 momentum=0.9,
                 eps=1e-5,
                 affine=True,
                 use_abn=False,
                 parallel=True):
        super(SepConv, self).__init__()
        if use_abn:
            self.op = nn.SequentialCell(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                          pad_mode='pad', padding=padding, group=in_channels,
                          has_bias=False, weight_init='HeNormal'),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, pad_mode='same',
                          has_bias=False, weight_init='HeNormal'),
                NormLeakyReLU(in_channels, momentum, eps, affine=affine, parallel=parallel),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                          pad_mode='pad', padding=padding, group=in_channels,
                          has_bias=False, weight_init='HeNormal'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='same',
                          has_bias=False, weight_init='HeNormal'),
                NormLeakyReLU(out_channels, momentum, eps, affine=affine, parallel=parallel)
            )

        else:
            self.op = nn.SequentialCell(
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                          pad_mode='pad', padding=padding, group=in_channels,
                          has_bias=False, weight_init='HeNormal'),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, pad_mode='same',
                          has_bias=False, weight_init='HeNormal'),
                BatchNormalization(in_channels, momentum, eps, affine=affine, parallel=parallel),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                          pad_mode='pad', padding=padding, group=in_channels,
                          has_bias=False, weight_init='HeNormal'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='same',
                          has_bias=False, weight_init='HeNormal'),
                BatchNormalization(out_channels, momentum, eps, affine=affine, parallel=parallel)
            )

    def construct(self, x):
        """construct"""
        return self.op(x)


class DilConv(nn.Cell):
    """Dilconv"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation=1,
                 momentum=0.9,
                 eps=1e-5,
                 affine=True,
                 separate=False,
                 use_abn=False,
                 parallel=True):
        super(DilConv, self).__init__()
        if use_abn:
            if separate:
                self.op = nn.SequentialCell(
                    nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                              pad_mode='pad', padding=padding, dilation=dilation, group=in_channels,
                              has_bias=False, weight_init='HeNormal'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='same',
                              has_bias=False, weight_init='HeNormal'),
                    NormLeakyReLU(out_channels, momentum, eps, affine=affine, parallel=parallel)
                )
            else:
                self.op = nn.SequentialCell(
                    nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                              pad_mode='pad', padding=padding, dilation=dilation,
                              has_bias=False, weight_init='HeNormal'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='same',
                              has_bias=False, weight_init='HeNormal'),
                    NormLeakyReLU(out_channels, momentum, eps, affine=affine, parallel=parallel)
                )

        else:
            if separate:
                self.op = nn.SequentialCell(
                    nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                              pad_mode='pad', padding=padding, dilation=dilation, group=in_channels,
                              has_bias=False, weight_init='HeNormal'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='same',
                              has_bias=False, weight_init='HeNormal'),
                    nn.BatchNorm2d(out_channels, affine=affine),
                )
            else:
                self.op = nn.SequentialCell(
                    nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                              pad_mode='pad', padding=padding, dilation=dilation,
                              has_bias=False, weight_init='HeNormal'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='same',
                              has_bias=False, weight_init='HeNormal'),
                    nn.BatchNorm2d(out_channels, affine=affine),
                )

    def construct(self, x):
        """construct"""
        return self.op(x)


class Identity(nn.Cell):
    """Identity"""
    def __init__(self):
        super(Identity, self).__init__()
        self.null = None

    def construct(self, x):
        """construct"""
        return x


class Zero(nn.Cell):
    """Zero"""
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.mul = ops.Mul()

    def construct(self, x):
        """construct"""
        if self.stride == 1:
            return self.mul(x, 0.)
        return self.mul(x[:, :, ::self.stride, ::self.stride], 0.)


class FactorizedReduce(nn.Cell):
    """FactorizedReduce"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 affine=True):
        super(FactorizedReduce, self).__init__()
        assert out_channels % 2 == 0
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2,
                                has_bias=False, pad_mode='valid', weight_init='HeNormal')
        self.conv_2 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2,
                                has_bias=False, pad_mode='valid', weight_init='HeNormal')
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def construct(self, x):
        """construct"""
        x = self.relu(x)
        cat = ops.Concat(axis=1)
        out = cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])])
        out = self.bn(out)
        return out


class ReLUConvBN(nn.Cell):
    """ReLUConvBN"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 pad_mode='pad',
                 momentum=0.9,
                 eps=1e-5,
                 affine=True,
                 use_abn=False,
                 parallel=True):
        super(ReLUConvBN, self).__init__()
        if use_abn:
            self.op = nn.SequentialCell(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                          pad_mode=pad_mode, padding=padding, has_bias=False),
                NormLeakyReLU(out_channels, momentum, eps, parallel=parallel)
            )

        else:
            self.op = nn.SequentialCell(
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                          pad_mode=pad_mode, padding=padding, has_bias=False),
                nn.BatchNorm2d(out_channels, momentum, eps, affine=affine)
            )

    def construct(self, x):
        """construct"""
        return self.op(x)
