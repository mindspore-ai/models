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
""" InstanceNorm2d """

import numpy as np
from mindspore import nn
from mindspore import Tensor, Parameter
from mindspore.common import initializer
import mindspore.ops.operations as P
import mindspore.ops.functional as F

class InstanceNorm2d(nn.Cell):
    """InstanceNorm2d"""

    def __init__(self, channel, affine=False):
        super(InstanceNorm2d, self).__init__()
        gamma_tensor = Tensor(np.ones(shape=[1, channel, 1, 1], dtype=np.float32))
        self.gamma = Parameter(initializer.initializer(
            init=gamma_tensor, shape=[1, channel, 1, 1]), name='gamma', requires_grad=affine)
        self.beta = Parameter(initializer.initializer(
            init=initializer.Zero(), shape=[1, channel, 1, 1]), name='beta', requires_grad=affine)
        self.reduceMean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sub = P.Sub()
        self.add = P.Add()
        self.rsqrt = P.Rsqrt()
        self.mul = P.Mul()
        self.tile = P.Tile()
        self.reshape = P.Reshape()
        self.eps = Tensor(np.ones(shape=[1, channel, 1, 1], dtype=np.float32) * 1e-5)
        self.cast2fp32 = P.Cast()

    def construct(self, x):
        mean = self.reduceMean(x, (2, 3))
        mean_stop_grad = F.stop_gradient(mean)
        variance = self.reduceMean(self.square(self.sub(x, mean_stop_grad)), (2, 3))
        variance = variance + self.eps
        inv = self.rsqrt(variance)
        normalized = self.sub(x, mean) * inv
        x_IN = self.add(self.mul(self.gamma, normalized), self.beta)
        return x_IN
