# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Define the operations."""
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops.operations as P
from mindspore.common import dtype as mstype

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, pad_mode='same'),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, pad_mode='same'),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(
            0, 3), has_bias=False, pad_mode='pad'),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(
            3, 0), has_bias=False, pad_mode='pad'),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Cell):
    """
    Define ReLUConvBN operatoin.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, has_bias=False, pad_mode='pad'),
            nn.BatchNorm2d(C_out, affine=affine)
        ])

    def construct(self, x):
        return self.op(x)


class DilConv(nn.Cell):
    """
    Define DilConv operatoin.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.dilation = dilation
        self.C_in = C_in
        self.op = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding // 2, group=C_in, has_bias=False, pad_mode='pad'),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0,
                      has_bias=False, pad_mode='pad'),
            nn.BatchNorm2d(C_out, affine=affine),
        ])

    def construct(self, x):
        return self.op(x)


class SepConv(nn.Cell):
    """
    Define SepConv operatoin.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, group=C_in, has_bias=False, pad_mode='pad'),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0,
                      has_bias=False, pad_mode='pad'),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, group=C_in, has_bias=False, pad_mode='pad'),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0,
                      has_bias=False, pad_mode='pad'),
            nn.BatchNorm2d(C_out, affine=affine),
        ])

    def construct(self, x):
        return self.op(x)


class Identity(nn.Cell):
    """
    Define Identity operatoin.
    """

    def __init__(self):
        super(Identity, self).__init__()
        print('Identity...')

    def construct(self, x):
        return x


class Zero(nn.Cell):
    """
    Define Zero operatoin.
    """

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def construct(self, x):
        n, c, h, w = x.shape
        h //= self.stride
        w //= self.stride
        padding = Tensor(np.zeros(shape=(n, c, h, w)), dtype=mindspore.float32)
        return padding


class FactorizedReduce(nn.Cell):
    """
    Define FactorizedReduce operatoin.
    """

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)
        self.conv_1 = nn.Conv2d(
            C_in, C_out // 2, 1, stride=2, padding=0, has_bias=False, pad_mode='pad')
        self.conv_2 = nn.Conv2d(
            C_in, C_out // 2, 1, stride=2, padding=0, has_bias=False, pad_mode='pad')
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def construct(self, x):
        x = self.relu(x)
        out = self.concat((self.conv_1(x), self.conv_2(x)))
        out = self.bn(out)
        return out


class GroupConv(nn.Cell):
    """
    group convolution operation.

    Args:
        in_channels (int): Input channels of feature map.
        out_channels (int): Output channels of feature map.
        kernel_size (int): Size of convolution kernel.
        stride (int): Stride size for the group convolution layer.

    Returns:
        tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_mode="pad",
                 padding=0, dilation=1, groups=1, has_bias=False):
        super(GroupConv, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups
        self.convs = nn.CellList()
        self.op_split = P.Split(axis=1, output_num=self.groups)
        self.op_concat = P.Concat(axis=1)
        self.cast = P.Cast()
        for _ in range(groups):
            self.convs.append(nn.Conv2d(in_channels // groups, out_channels // groups,
                                        kernel_size=kernel_size, stride=stride, has_bias=has_bias,
                                        padding=padding, pad_mode=pad_mode, dilation=dilation, group=1))

    def construct(self, x):
        features = self.op_split(x)
        outputs = ()
        for i in range(self.groups):
            outputs = outputs + \
                (self.convs[i](self.cast(features[i], mstype.float32)),)
        out = self.op_concat(outputs)
        return out
