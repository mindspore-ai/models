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
"""
Basic layers
"""

import mindspore.nn as nn
from src.init_weights import KaimingUniform, UniformBias


class Conv2dLikeTorch(nn.Conv2d):
    """Conv2dTransposeLikeTorch"""
    def __init__(self, in_channel, out_channel, kernel_size, pad_mode, padding, stride, has_bias):
        initializer = KaimingUniform()
        bias_initializer = UniformBias(shape=(out_channel, in_channel, kernel_size, kernel_size), mode='fan_in')
        super().__init__(in_channel, out_channel, kernel_size, pad_mode=pad_mode, weight_init=initializer,
                         padding=padding, stride=stride, has_bias=has_bias, bias_init=bias_initializer)


class Conv2dTransposeLikeTorch(nn.Conv2dTranspose):
    """Conv2dTransposeLikeTorch"""
    def __init__(self, in_channel, out_channel, kernel_size, pad_mode, padding, stride, has_bias):
        initializer = KaimingUniform(mode='fan_in')
        bias_initializer = UniformBias(shape=(out_channel, in_channel, kernel_size, kernel_size), mode='fan_in')
        super().__init__(in_channel, out_channel, kernel_size, pad_mode=pad_mode, weight_init=initializer,
                         padding=padding, stride=stride, has_bias=has_bias, bias_init=bias_initializer)


class BasicConv(nn.Cell):
    """basic conv block"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super().__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                Conv2dTransposeLikeTorch(in_channel, out_channel, kernel_size, pad_mode='pad',
                                         padding=padding, stride=stride, has_bias=bias)
            )
        else:
            layers.append(
                Conv2dLikeTorch(in_channel, out_channel, kernel_size, pad_mode='pad',
                                padding=padding, stride=stride, has_bias=bias)
            )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU())
        self.main = nn.SequentialCell(layers)

    def construct(self, x):
        """construct basic conv block"""
        return self.main(x)


class ResBlock(nn.Cell):
    """residual block"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.main = nn.SequentialCell(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def construct(self, x):
        """construct residual block"""
        return self.main(x) + x
