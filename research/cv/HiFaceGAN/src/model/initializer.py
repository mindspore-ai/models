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
"""Initializer for neural networks"""
import math

import numpy as np
from mindspore import nn
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Initializer
from mindspore.common.initializer import Normal

import src.model.architecture as arch


def _assignment(arr, num):
    """Assign the value of `num` to `arr`"""
    if arr.shape == ():
        arr = arr.reshape(1)
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr


def _calculate_fan_in_and_fan_out(shape):
    """Calculate fan_in and fan_out"""
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("'fan_in' and 'fan_out' can not be computed for tensor with fewer than"
                         " 2 dimensions, but got dimensions {}.".format(dimensions))
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        for i in range(2, dimensions):
            receptive_field_size *= shape[i]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


class XavierNormal(Initializer):
    """Xavier Normal initializer"""

    def __init__(self, gain=0.02):
        super().__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        """Initialize tensor using Xavier Normal distribution"""
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)

        sigma = self.gain * math.sqrt(2.0 / (n_in + n_out))
        data = np.random.normal(0, sigma, arr.shape)

        _assignment(arr, data)


def init_weights(net):
    """Initialize weights of a neural network"""
    for _, cell in net.cells_and_names():

        if isinstance(cell, nn.BatchNorm2d):
            # if affine is True
            if cell.gamma.requires_grad:
                cell.gamma.set_data(initializer(Normal(sigma=0.02, mean=1.0), cell.gamma.shape,
                                                cell.gamma.dtype))
        elif isinstance(cell, arch.Conv2dNormalized):
            cell.weight_orig.set_data(initializer(XavierNormal(gain=0.02), cell.weight_orig.shape,
                                                  cell.weight_orig.dtype))
        elif isinstance(cell, (nn.Conv2d, nn.Dense)):
            cell.weight.set_data(initializer(XavierNormal(gain=0.02), cell.weight.shape,
                                             cell.weight.dtype))
