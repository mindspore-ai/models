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
""" Spectral_norm """

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.layer.conv import _check_input_3d

stdnormal = ops.StandardNormal(seed=43)
l2normalize = ops.L2Normalize(epsilon=1e-12)


class SpectualNormConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCHW',
                 power_iterations=1):
        super(SpectualNormConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init,
            data_format)
        self.power_iterations = power_iterations
        height = self.weight.shape[0]
        width = self.weight.view(height, -1).shape[1]
        self.weight_u = mindspore.Parameter(l2normalize(stdnormal((height, 1))), requires_grad=False)
        self.weight_v = mindspore.Parameter(l2normalize(stdnormal((width, 1))), requires_grad=False)

    def construct(self, x):
        height = self.weight.shape[0]
        for _ in range(self.power_iterations):
            self.weight_v = l2normalize(ops.tensor_dot(self.weight.view(height, -1).T, self.weight_u, axes=1))
            self.weight_u = l2normalize(ops.tensor_dot(self.weight.view(height, -1), self.weight_v, axes=1))
        sigma = ops.tensor_dot(self.weight_u.T, self.weight.view(height, -1), axes=1)
        sigma = ops.tensor_dot(sigma, self.weight_v, axes=1)
        weight = self.weight / sigma
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


class SpectualNormConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCHW',
                 power_iterations=1):
        super(SpectualNormConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init)
        self.power_iterations = power_iterations
        height = self.weight.shape[0]
        width = self.weight.view(height, -1).shape[1]
        self.weight_u = mindspore.Parameter(l2normalize(stdnormal((height, 1))), requires_grad=False)
        self.weight_v = mindspore.Parameter(l2normalize(stdnormal((width, 1))), requires_grad=False)

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_3d(x_shape)
        x = self.expand_dims(x, 2)
        height = self.weight.shape[0]
        for _ in range(self.power_iterations):
            self.weight_v = l2normalize(ops.tensor_dot(self.weight.view(height, -1).T, self.weight_u, axes=1))
            self.weight_u = l2normalize(ops.tensor_dot(self.weight.view(height, -1), self.weight_v, axes=1))
        sigma = ops.tensor_dot(self.weight_u.T, self.weight.view(height, -1), axes=1)
        sigma = ops.tensor_dot(sigma, self.weight_v, axes=1)
        weight = self.weight / sigma
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        output = self.squeeze(output)
        return output
