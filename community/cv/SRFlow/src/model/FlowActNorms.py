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
The Actnorm Layer of Glow
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.ops import stop_gradient


class ActNormNotHasLogdet(nn.Cell):
    """
    In Graph_mode, logdet can not be none,
    So the class of Actnorm's not has logdet
    The train part of Actnorm not has logdet
    """
    def __init__(self, num_features):
        super().__init__()

        zeros = ops.Zeros()
        zero = zeros((1, num_features, 1, 1), mindspore.float32)

        self.bias = Parameter(default_input=zero, name='bias')
        self.logs = Parameter(default_input=zero, name='logs')

    def construct(self, x):
        exp = ops.Exp()
        x = (x + self.bias) * exp(self.logs)

        return x


class ActNormHasLogdet(nn.Cell):
    """
    In Graph_mode, logdet can not be none,
    So the class of Actnorm's has logdet
    The train part of Actnorm has logdet
    """
    def __init__(self, num_features):
        super().__init__()

        zeros = ops.Zeros()
        zero = zeros((1, num_features, 1, 1), mindspore.float32)

        ones = ops.Ones()
        ones = ones((1, num_features, 1, 1), mindspore.float32)

        self.bias = Parameter(default_input=ones / 1e3, name='bias')
        self.logs = Parameter(default_input=zero, name='logs')

    def construct(self, x, logdet=None):
        exp = ops.Exp()
        reduce_sum = ops.ReduceSum()
        shape = ops.Shape()
        x = (x + self.bias) * exp(self.logs)
        logdet += reduce_sum(self.logs) * shape(x)[2] * shape(x)[3]
        return x, logdet


class ActNormHasLogdetRev(nn.Cell):
    """
    The test part of Actnorm has logdet
    """
    def __init__(self, num_features):
        super().__init__()

        zeros = ops.Zeros()
        zero = zeros((1, num_features, 1, 1), mindspore.float32)

        self.bias = Parameter(default_input=zero, name='bias')
        self.logs = Parameter(default_input=zero, name='logs')

    def construct(self, x, logdet=None):
        exp = ops.Exp()
        reduce_sum = ops.ReduceSum()
        shape = ops.Shape()
        self.bias = stop_gradient(self.bias)
        self.logs = stop_gradient(self.logs)
        x = x * exp(-self.logs)
        x -= reduce_sum(self.logs) * shape(x)[2] * shape(x)[3]
        x = x - self.bias
        return x, logdet
