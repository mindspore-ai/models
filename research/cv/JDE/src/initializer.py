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
# This file was copied from project [ascend][modelzoo-his]
import math
from functools import reduce
import numpy as np

from mindspore.common import initializer as init
from mindspore.common.initializer import Initializer as MeInitializer

def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    if nonlinearity == 'tanh':
        return 5.0 / 3
    if nonlinearity == 'relu':
        return math.sqrt(2.0)
    if nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))

    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _assignment(arr, num):
    """Assign the value of 'num' and 'arr'."""
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr

def _calculate_correct_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(array)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform_(arr, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(arr, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, arr.shape)

def _calculate_fan_in_and_fan_out(arr):
    """Calculate fan in and fan out."""
    dimensions = len(arr.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for array with fewer than 2 dimensions")

    num_input_fmaps = arr.shape[1]
    num_output_fmaps = arr.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = reduce(lambda x, y: x * y, arr.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

class KaimingUniform(MeInitializer):
    """Kaiming uniform initializer."""
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingUniform, self).__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def _initialize(self, arr):
        tmp = kaiming_uniform_(arr, self.a, self.mode, self.nonlinearity)
        _assignment(arr, tmp)

def init_cov(conv):
    conv.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                          conv.weight.shape,
                                          conv.weight.dtype))
    if conv.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(conv.weight)
        bound = 1 / math.sqrt(fan_in)
        conv.bias.set_data(init.initializer(init.Uniform(bound),
                                            conv.bias.shape,
                                            conv.bias.dtype))
    return conv

def init_bn(after_bn):
    scale = 0.1
    after_bn.gamma.set_data(init.initializer(init.Uniform(scale),
                                             after_bn.gamma.shape,
                                             after_bn.gamma.dtype))

def init_dense(classifier):
    classifier.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                classifier.weight.shape,
                                                classifier.weight.dtype))
    if classifier.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(classifier.weight)
        bound = 1 / math.sqrt(fan_in)
        classifier.bias.set_data(init.initializer(init.Uniform(bound),
                                                  classifier.bias.shape,
                                                  classifier.bias.dtype))
    return classifier
