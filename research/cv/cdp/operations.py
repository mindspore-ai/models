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

import mindspore.nn as nn
import mindspore.ops as ops

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
    'conv_7x1_1x7': lambda C, stride, affine: nn.SequentialCell(
        nn.ReLU(),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), pad_mode="pad", padding=(0, 3), has_bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), pad_mode="pad", padding=(3, 0), has_bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Cell):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, pad_mode="pad", padding=padding, has_bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def construct(self, x):
        return self.op(x)


class DilConv(nn.Cell):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, pad_mode="pad",
                      padding=padding, dilation=dilation, group=C_in, has_bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, pad_mode="pad", padding=0, has_bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def construct(self, x):
        return self.op(x)

class SepConv(nn.Cell):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      pad_mode="pad", padding=padding, group=C_in, has_bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, pad_mode="pad", padding=0, has_bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, pad_mode="pad",
                      padding=padding, group=C_in, has_bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, pad_mode="pad", padding=0, has_bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def construct(self, x):
        return self.op(x)

class Identity(nn.Cell):

    def construct(self, x):
        return x

class Zero(nn.Cell):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def construct(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Cell):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, has_bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, has_bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        x = self.relu(x)
        out = self.concat((self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])))
        out = self.bn(out)
        return out
