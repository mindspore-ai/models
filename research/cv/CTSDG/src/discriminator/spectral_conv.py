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
"""spectral conv"""

import numpy as np
from mindspore import Parameter
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Normal
from mindspore.common.initializer import initializer


class Conv2dNormalized(nn.Cell):
    """Conv2d layer with spectral normalization"""
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            has_bias=False,
            pad=0,
            pad_mode="same",
    ):
        super().__init__()
        self.conv2d = ops.Conv2D(out_channel=out_channel, kernel_size=kernel_size, stride=stride,
                                 mode=1, pad=pad, pad_mode=pad_mode)
        self.bias_add = ops.BiasAdd(data_format="NCHW")
        self.has_bias = has_bias

        if self.has_bias:
            self.bias = Parameter(initializer('zeros', (out_channel,)), name='bias')

        self.weight_orig = Parameter(
            initializer(Normal(sigma=0.02), (out_channel, in_channel, kernel_size, kernel_size)),
            name='weight_orig'
        )

        self.weight_u = Parameter(self.initialize_param(out_channel, 1), requires_grad=False, name='weight_u')
        self.weight_v = Parameter(self.initialize_param(in_channel * kernel_size * kernel_size, 1), requires_grad=False,
                                  name='weight_v')

    @staticmethod
    def initialize_param(*param_shape):
        """initialize params"""
        param = np.random.randn(*param_shape).astype('float32')
        return param / np.linalg.norm(param)

    def normalize_weights(self, weight_orig, u, v):
        """Weights normalization"""
        size = weight_orig.shape
        weight_mat = weight_orig.ravel().view(size[0], -1)

        v = ops.matmul(weight_mat.T, u)
        v_norm = nn.Norm()(v)
        v = v / v_norm

        u = ops.matmul(weight_mat, v)
        u_norm = nn.Norm()(u)
        u = u / u_norm

        u = ops.depend(u, ops.assign(self.weight_u, u))
        v = ops.depend(v, ops.assign(self.weight_v, v))

        u = ops.stop_gradient(u)
        v = ops.stop_gradient(v)

        weight_norm = ops.matmul(u.T, ops.matmul(weight_mat, v))
        weight_sn = weight_mat / weight_norm
        weight_sn = weight_sn.view(*size)

        return weight_sn

    def construct(self, x):
        """Feed forward"""
        weight = self.normalize_weights(self.weight_orig, self.weight_u, self.weight_v)
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output
