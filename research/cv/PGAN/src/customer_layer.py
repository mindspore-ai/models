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
"""cell define"""
import math
import mindspore
from mindspore import ops, nn
import numpy as np
from numpy import prod

def num_flat_features(x):
    return x.size//x.shape[0]
def getLayerNormalizationFactor(x):
    """
    Get He's constant for the given layer

    Returns:
        output.
    """
    size = x.weight.shape
    fan_in = prod(size[1:])
    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Cell):
    """
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initBiasToZero=True):
        super(ConstrainedLayer, self).__init__()
        self.module = module
        self.equalized = equalized
        if initBiasToZero:
            bias_shape = self.module.bias.shape
            zeros = ops.Zeros()
            self.module.bias.set_data(zeros(bias_shape, mindspore.float32))
        if self.equalized:
            weight_shape = self.module.weight.shape
            wight_init = np.random.normal(loc=0.0, scale=1.0, size=weight_shape) / lrMul
            self.module.weight.set_data(mindspore.Tensor(wight_init, mindspore.float32))
            self.lr_weight = getLayerNormalizationFactor(self.module) * lrMul

    def construct(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.lr_weight
        return x


class EqualizedLinear(ConstrainedLayer):
    """
    EqualizedLinear
    """
    def __init__(self,
                 nChannelsPrevious,
                 nChannels):
        """
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
        """
        ConstrainedLayer.__init__(self,
                                  nn.Dense(nChannelsPrevious, nChannels))


class EqualizedConv2d(ConstrainedLayer):
    """
    EqualizedConv2d
    """
    def __init__(self, depthNewScale, out, kernnel, padding, pad_mode="pad", has_bias=True):
        """
        A nn.Conv2d module with specific constraints
        Args:
            depthNewScale (int): number of channels in the previous layer
            out (int): number of channels of the current layer
            kernnel (int): size of the convolutional kernel
            padding (int): convolution's padding
            has_bias (bool): with bias ?
        """
        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(depthNewScale, out, kernnel, padding=padding, pad_mode=pad_mode,
                                            has_bias=has_bias))
