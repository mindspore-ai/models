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
"""
python initializer.py
"""
import math
import numpy as np

from mindspore.common.initializer import Initializer, _calculate_fan_in_and_fan_out, _assignment, _register

_INITIALIZER_ALIAS = dict()

@_register('glorot_normal')
class GlorotNormal(Initializer):
    r"""
    Initialize the array with Glorot Normal algorithm, and from a normal distribution collect samples within
    N(0, std), The std is defined as:

    .. math::
        std = gain * \sqrt{\frac{2}{n_{in} + n_{out}}}

    - where :math:`n_{in}` is the number of input units in the weight tensor.
    - where :math:`n_{out}` is the number of output units in the weight tensor.

    Args:
        gain (float): An optional scaling factor. Default: 1.

    Returns:
        Array, assigned array.
    """

    def __init__(self, gain=1):
        super(GlorotNormal, self).__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)

        std = self.gain * math.sqrt(2.0 / (n_in + n_out))
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)
