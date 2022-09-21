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
"""FPN"""
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import ResizeNearestNeighbor
import mindspore
from mindspore.common.initializer import initializer, HeUniform
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
import numpy as np


def bias_init_zeros(shape):
    """Bias init method."""
    return Tensor(np.array(np.zeros(shape).astype(np.float32)))


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer(HeUniform(negative_slope=1), shape=shape, dtype=mstype.float32).to_tensor()
    shape_bias = (out_channels,)
    biass = bias_init_zeros(shape_bias)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)

class FPN(nn.Cell):
    '''only for resnet50,101,152'''

    def __init__(self, features=256, use_p5=True):
        super(FPN, self).__init__()

        self.prj_5 = _conv(2048, features, kernel_size=1, stride=1, pad_mode='valid')
        self.prj_4 = _conv(1024, features, kernel_size=1, stride=1, pad_mode='valid')
        self.prj_3 = _conv(512, features, kernel_size=1, pad_mode='valid')
        self.conv_5 = _conv(features, features, kernel_size=3, pad_mode='pad', padding=1)
        self.conv_4 = _conv(features, features, kernel_size=3, pad_mode='pad', padding=1)
        self.conv_3 = _conv(features, features, kernel_size=3, pad_mode='pad', padding=1)
        if use_p5:
            self.conv_out6 = _conv(features, features, kernel_size=3, pad_mode='pad', padding=1, stride=2)
        else:
            self.conv_out6 = _conv(2048, features, kernel_size=3, pad_mode='pad', padding=1, stride=2)
        self.conv_out7 = _conv(features, features, kernel_size=3, pad_mode='pad', padding=1, stride=2)
        self.use_p5 = use_p5
        constant_init = mindspore.common.initializer.Constant(0)
        constant_init(self.prj_5.bias)
        constant_init(self.prj_4.bias)
        constant_init(self.prj_3.bias)
        constant_init(self.conv_5.bias)
        constant_init(self.conv_4.bias)
        constant_init(self.conv_3.bias)
        constant_init(self.conv_out6.bias)
        constant_init(self.conv_out7.bias)

    def upsamplelike(self, inputs):
        src, target = inputs
        resize = ResizeNearestNeighbor((target.shape[2], target.shape[3]))
        return resize(src)

    def construct(self, x):
        C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        P4 = P4 + self.upsamplelike((P5, C4))
        P3 = P3 + self.upsamplelike((P4, C3))
        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        relu = ops.ReLU()
        P7 = self.conv_out7(relu(P6))
        return (P3, P4, P5, P6, P7)
