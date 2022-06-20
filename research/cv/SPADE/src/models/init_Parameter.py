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
""" Xavier Normal """

import math
import numpy as np


class XavierNormal:
    def __init__(self, gain=1):
        self.gain = gain

    def initialize(self, shape):
        n_in, n_out = _calculate_fan_in_and_fan_out(shape)
        std = self.gain * math.sqrt(2.0 / (n_in + n_out))
        data = np.random.normal(0, std, shape)
        return data


def _calculate_fan_in_and_fan_out(shape):
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
