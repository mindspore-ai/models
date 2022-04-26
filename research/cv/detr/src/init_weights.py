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
"""init weights"""
import math

import numpy as np
from mindspore.common import initializer as init
from mindspore.common.initializer import _assignment
from mindspore.common.initializer import _calculate_correct_fan
from mindspore.common.initializer import _calculate_fan_in_and_fan_out
from mindspore.common.initializer import _calculate_gain


class KaimingUniform(init.Initializer):
    """
    Initialize the array with He kaiming algorithm.
    Args:
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function, recommended to use only with
            ``'relu'`` or ``'leaky_relu'`` (default).
    """
    def __init__(self, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'):
        super().__init__()
        self.mode = mode
        self.gain = _calculate_gain(nonlinearity, a)

    def _initialize(self, arr):
        fan = _calculate_correct_fan(arr.shape, self.mode)
        bound = math.sqrt(3.0) * self.gain / math.sqrt(fan)
        data = np.random.uniform(-bound, bound, arr.shape)
        _assignment(arr, data)


class UniformBias(init.Initializer):
    """bias uniform initializer"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def _initialize(self, arr):
        fan_in, _ = _calculate_fan_in_and_fan_out(self.shape)
        bound = 1 / math.sqrt(fan_in)
        data = np.random.uniform(-bound, bound, arr.shape)
        _assignment(arr, data)
