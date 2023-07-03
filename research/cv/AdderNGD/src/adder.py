# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
import res20_adder


def unfold(img, kernel_size, stride=1, pad=0, dilation=1):
    """
    unfold function
    """
    batch_num, channel, height, width = img.shape
    out_h = (height + pad + pad - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1
    out_w = (width + pad + pad - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1

    img = np.pad(img, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((batch_num, channel, kernel_size, kernel_size, out_h, out_w)).astype(img.dtype)

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = np.reshape(col, (batch_num, channel * kernel_size * kernel_size, out_h * out_w))

    return col


class Adder(nn.Cell):
    """
    Adder operation
    """
    def __init__(self):
        super(Adder, self).__init__()
        self.abs = ops.Abs()
        self.sum = ops.ReduceSum(keep_dims=False)
        self.expand_dims = ops.ExpandDims()
        self.lp_norm = ops.LpNorm(axis=[0, 1], p=2, keep_dims=False)
        self.sqrt = ops.Sqrt()

    def construct(self, w_col, x_col):
        output = -self.sum(self.abs((self.expand_dims(w_col, 2)-self.expand_dims(x_col, 0))), 1)
        return output


def adder2d_function(x, w, stride=1, padding=0):
    """
    Adder2d function
    """
    n_filters, _, h_filter, w_filter = w.shape
    n_x, _, h_x, w_x = x.shape

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    # here rates padding
    x_col = unfold((x.view(1, -1, h_x, w_x)), kernel_size=h_filter,
                   stride=stride, pad=padding, dilation=1).view(n_x, -1, h_out * w_out)
    adder = Adder()

    #here ravel
    x_col = ops.transpose(x_col, (1, 2, 0)).view(x_col.shape[1], -1)
    w_col = w.view(n_filters, -1)

    out = adder(w_col, x_col)
    out = out.view(n_filters, h_out, w_out, n_x)
    out = ops.transpose(out, (3, 0, 1, 2))
    return out


class Adder2d(nn.Cell):
    def __init__(self, in_channels, output_channel, kernel_size, stride=1,
                 padding=0, bias=False):
        super(Adder2d, self).__init__()
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.output_channel = output_channel
        self.kernel_size = kernel_size

        weight_shape = (output_channel, in_channels, kernel_size, kernel_size)
        weight = Tensor(res20_adder.kaiming_normal(weight_shape,
                                                   mode="fan_out", nonlinearity='relu'))
        self.adder = Parameter(weight)
        self.bias = bias
        if bias:
            b_weight = Tensor(res20_adder.kaiming_normal((output_channel),
                                                         mode="fan_out", nonlinearity='relu'))
            self.b = Parameter(b_weight)

    def construct(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output
