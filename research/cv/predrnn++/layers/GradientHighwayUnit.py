# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.initializer import Uniform
from mindspore import Tensor

from layers.LayerNorm import LayerNorm

EPSILON = 0.00001

def init_state(inputs, num_features):
    dims = inputs.get_shape().ndims
    if dims == 4:
        batch = inputs.get_shape()[0]
        height = inputs.get_shape()[1]
        width = inputs.get_shape()[2]
    else:
        raise ValueError('input tensor should be rank 4.')
    z_init = Tensor(np.zeros((batch, num_features, height, width)).astype(np.float32))
    return z_init

class GHU(nn.Cell):
    def __init__(self, layer_name, input_shape, filter_size, num_hidden_0, tln=False,
                 init_value=0.001):
        """Initialize the Gradient Highway Unit.
        """
        super(GHU, self).__init__()
        self.ghu_name = layer_name
        self.num_features = num_hidden_0
        self.layer_norm = tln
        self.shape = input_shape

        if init_value == -1:
            self.initializer = None
        else:
            self.initializer = Uniform(init_value)

        self.z_conv = nn.Conv2d(self.num_features, self.num_features*2, filter_size, 1, padding=0, pad_mode='same',
                                has_bias=True, weight_init=self.initializer)

        self.x_conv = nn.Conv2d(self.num_features, self.num_features*2, filter_size, 1, padding=0, pad_mode='same',
                                has_bias=True, weight_init=self.initializer)

        self.layer_norm_z = LayerNorm(self.num_features*2)
        self.layer_norm_x = LayerNorm(self.num_features*2)

        self.add = P.TensorAdd()

        self.split = P.Split(1, 2)

        self.tanh = nn.Tanh()

        self.sigmoid = nn.Sigmoid()

        self.zeros = P.Zeros()

    def construct(self, x, z):
        if z is None:
            z = self.zeros((self.shape[0], self.num_features, self.shape[3], self.shape[4]), ms.float32)

        z_out = self.z_conv(z)
        if self.layer_norm:
            z_out = self.layer_norm_z(z_out)

        out_x = self.x_conv(x)
        if self.layer_norm:
            out_x = self.layer_norm_x(out_x)

        gates = self.add(out_x, z_out)

        p, u = self.split(gates)
        p = self.tanh(p)
        u = self.sigmoid(u)

        z_new = u * p + (1-u) * z
        return z_new
