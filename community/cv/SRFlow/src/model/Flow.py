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
It Contains Conv2D and Squeeze Layer of Glow
"""

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from mindspore import Parameter

from src.model.FlowActNorms import ActNormNotHasLogdet


class Conv2d(nn.Conv2d):
    """
    Custom Cond2D, weight_init is normal
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, weight_init='normal',
                 pad_mode='pad', padding=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, weight_init=weight_init,
                         has_bias=False, pad_mode=pad_mode, padding=padding)

        self.actnorm = ActNormNotHasLogdet(out_channels)

    def construct(self, x):
        x = super().construct(x)
        x = self.actnorm(x)

        return x


class Conv2dZeros(nn.Conv2d):
    """
    Custom Cond2D, weight_init is zero
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, weight_init='zeros',
                 bias_init='zeros', logscale_factor=3, pad_mode='pad', padding=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, weight_init=weight_init,
                         bias_init=bias_init, has_bias=True, pad_mode=pad_mode, padding=padding)

        self.logscale_factor = logscale_factor

        zeros = ops.Zeros()
        self.logs = Parameter(default_input=zeros((out_channels, 1, 1), mindspore.float32), name='logs')

    def construct(self, x):
        x = super().construct(x)
        exp = ops.Exp()
        x = x * exp(self.logs * self.logscale_factor)

        return x


class SqueezeLayerFor(nn.Cell):
    """
    The train part of Squeeze Layer
    """
    def __init__(self, factor):
        super().__init__()

        self.factor = factor

    def construct(self, x, logdet=None, rrdbResults=None):
        x = self.squeeze2d(x, self.factor)
        return x, logdet

    def squeeze2d(self, x, factor=2):
        shape = ops.Shape()
        B, C, H, W = shape(x)
        reshape = ops.Reshape()
        transpose = ops.Transpose()
        x = reshape(x, (B, C, H // factor, factor, W // factor, factor))
        x = transpose(x, (0, 1, 3, 5, 2, 4))
        x = reshape(x, (B, C * factor * factor, H // factor, W // factor))
        return x


class SqueezeLayerRev(nn.Cell):
    """
    The test part of Squeeze Layer
    """
    def __init__(self, factor):
        super().__init__()

        self.factor = factor

    def construct(self, x, logdet=None, rrdbResults=None):
        x = self.unsqueeze2d(x, self.factor)
        return x, logdet

    def unsqueeze2d(self, x, factor=2):
        factor2 = factor ** 2
        shape = ops.Shape()
        B, C, H, W = shape(x)
        reshape = ops.Reshape()
        transpose = ops.Transpose()
        x = reshape(x, (B, C // factor2, factor, factor, H, W))
        x = transpose(x, (0, 1, 4, 2, 5, 3))
        x = reshape(x, (B, C // factor2, H * factor, W * factor))
        return x
