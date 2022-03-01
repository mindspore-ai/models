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
"""IsyNet layers implementation"""
import numpy as np
from scipy.stats import truncnorm

from mindspore import nn
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore import Parameter
import mindspore.common.dtype as mstype


__all__ = ['Identity', 'Conv2dBatchActivation', 'ResidualCell', 'conv2d_custom']

class NoSkipCell(nn.Cell):
    """Identity Cell"""
    def construct(self, *inputs, **_kwargs):
        """construct"""
        x, _ = inputs
        return x

class ResidualCell(nn.Cell):
    """Residual Cell with flexible skip"""
    def __init__(self, content_cell, op_type):
        super().__init__()
        self.content_cell = content_cell
        self.op_type = op_type
        if not op_type in ["noSkip", "add", "concat"]:
            raise NotImplementedError("Operation " + op_type + " is not implemented for ResidualCell")
        if op_type == "noSkip":
            self.op = NoSkipCell()
        elif op_type == "add":
            self.op = ops.Add()
        elif op_type == "concat":
            self.op = ops.Concat(axis=1) # concatenate channels

    def __repr__(self):
        s = 'ResidualCell({content_cell}, {op_type})'
        return s.format(content_cell=self.content_cell, op_type=self.op_type)

    def construct(self, *inputs, **_kwargs):
        x = inputs[0]
        y = self.content_cell(x)
        return self.op(y, x)

def conv2d_custom(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  groups,
                  weight_standardization=0):
    """Standard convolution or Convolution with weight standardization"""
    if weight_standardization:
        return Conv2dWS(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding, groups=groups)
    return Conv2dStandard(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, groups=groups)

class Conv2dBatchActivation(nn.Cell):
    """
       Conv2d->BatchNorm->Activation sequence
    """
    def __init__(self, operation, group, stride, in_channels, out_channels, activation, weight_standardization=0):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.group = group
        self.activation = activation
        self.operation = operation
        self.weight_standardization = weight_standardization
        kernel_size = 0
        if operation == 'conv_1x1': kernel_size = 1
        if operation == 'conv_3x3': kernel_size = 3
        if operation == 'conv_5x5': kernel_size = 5
        if operation == 'conv_7x7': kernel_size = 7
        if operation == 'conv_1x3_3x1': kernel_size = (3, 1)
        if operation == 'conv_1x5_5x1': kernel_size = (5, 1)
        if operation == 'conv_1x7_7x1': kernel_size = (7, 1)
        self.kernel_size = kernel_size
        if isinstance(kernel_size, (list, tuple)):
            self.padding1 = (int(kernel_size[0]/2), int(kernel_size[1]/2))
            self.padding2 = (int(kernel_size[1]/2), int(kernel_size[0]/2))
            self.padding = 0
        else:
            self.padding = int(kernel_size/2)

        # Layer
        if operation != 'identity':
            if isinstance(kernel_size, (list, tuple)) and self.activation == 'Relu':
                self.layers = nn.SequentialCell([
                    conv2d_custom(self.in_channels,
                                  self.out_channels,
                                  (kernel_size[0], kernel_size[1]),
                                  (self.stride, 1),
                                  self.padding1,
                                  groups=self.group,
                                  weight_standardization=self.weight_standardization),
                    conv2d_custom(self.out_channels,
                                  self.out_channels,
                                  (kernel_size[1], kernel_size[0]),
                                  (1, self.stride),
                                  self.padding2,
                                  groups=self.group,
                                  weight_standardization=self.weight_standardization),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU()
                ])
            if isinstance(kernel_size, (list, tuple)) and self.activation != 'Relu':
                self.layers = nn.SequentialCell([
                    conv2d_custom(self.in_channels,
                                  self.out_channels,
                                  (kernel_size[0], kernel_size[1]),
                                  (self.stride, 1),
                                  self.padding1,
                                  groups=self.group,
                                  weight_standardization=self.weight_standardization),
                    conv2d_custom(self.out_channels,
                                  self.out_channels,
                                  (kernel_size[1], kernel_size[0]),
                                  (1, self.stride),
                                  self.padding2,
                                  groups=self.group,
                                  weight_standardization=self.weight_standardization),
                    nn.BatchNorm2d(self.out_channels)
                ])
            if not(isinstance(kernel_size, (list, tuple))) and self.activation == 'Relu':
                self.layers = nn.SequentialCell([
                    conv2d_custom(self.in_channels,
                                  self.out_channels,
                                  kernel_size,
                                  self.stride,
                                  self.padding,
                                  groups=self.group,
                                  weight_standardization=self.weight_standardization),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU()
                ])
            if not(isinstance(kernel_size, (list, tuple))) and self.activation != 'Relu':
                self.layers = nn.SequentialCell([
                    conv2d_custom(self.in_channels,
                                  self.out_channels,
                                  kernel_size,
                                  self.stride,
                                  self.padding,
                                  groups=self.group,
                                  weight_standardization=self.weight_standardization),
                    nn.BatchNorm2d(self.out_channels)
                ])

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def construct(self, *inputs, **_kwargs):
        x = inputs[0]
        if self.operation == 'identity':
            return x
        return self.layers(x)

def _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    """Initialization for convolution operation"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    fan_in = in_channel * kernel_size[0] * kernel_size[1]
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale**0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size[0] * kernel_size[1])
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size[0], kernel_size[1]))
    return Tensor(weight, dtype=mstype.float32)

class WeightStandardization(nn.Cell):
    """Weight standardization"""
    def __init__(self, eps=5e-4):
        super().__init__()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.square = ops.Square()
        self.sqrt = ops.Sqrt()
        self.eps = eps

    def construct(self, *inputs, **_kwargs):
        x = inputs[0]
        # weights channels are (Cout, Cin, K1, K2)
        x = x - self.mean(self.mean(self.mean(x, 1), 2), 3)
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        n = c*h*w
        std = self.sqrt((n/(n-1))*self.mean(self.square(x), (1, 2, 3))) + self.eps
        return x / std

class Conv2dWS(nn.Cell):
    """Convolution with weight standardization"""
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, pad_mode="pad", groups=1, weight_init=None):
        super().__init__()
        if isinstance(padding, tuple) and len(padding) == 2:
            px, py = padding
            padding = (px, px, py, py)
        if weight_init is None:
            weight_init = _conv_variance_scaling_initializer(in_channels, out_channels, kernel_size)
        self.conv_2d = ops.Conv2D(out_channel=out_channels, kernel_size=kernel_size, mode=1,
                                  pad_mode=pad_mode, pad=padding, stride=stride, group=groups)
        self.weight = Parameter(weight_init, name="weight")
        self.use_bias = False
        self.norm_op = WeightStandardization()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.sub = ops.Sub()

    def construct(self, *inputs, **_kwargs):
        x = inputs[0]
        weight = self.norm_op(self.weight)
        y = self.conv_2d(x, weight)
        return y

class Conv2dStandard(nn.Cell):
    """Standard convolution """
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, pad_mode="pad", groups=1, weight_init=None):
        super().__init__()
        if isinstance(padding, tuple) and len(padding) == 2:
            px, py = padding
            padding = (px, px, py, py)
        if weight_init is None:
            weight_init = _conv_variance_scaling_initializer(in_channels, out_channels, kernel_size)
        self.conv_2d = ops.Conv2D(out_channel=out_channels, kernel_size=kernel_size, mode=1,
                                  pad_mode=pad_mode, pad=padding, stride=stride, group=groups)
        self.weight = Parameter(weight_init, name="weight")

    def construct(self, *inputs, **_kwargs):
        x = inputs[0]
        y = self.conv_2d(x, self.weight)
        return y

class GlobalAvgPool2d(nn.Cell):
    """ This layer averages each channel to a single number.
    """
    def __init__(self, keep_dims=False):
        super().__init__()
        self.keep_dims = keep_dims
        self.mean = ops.ReduceMean(keep_dims=keep_dims)

    def construct(self, *inputs, **_kwargs):
        x = inputs[0]
        return self.mean(x, (1, 2))

class Identity(nn.Cell):
    """Identity cell"""
    def construct(self, *inputs, **_kwargs):
        x = inputs[0]
        return x

    def __repr__(self):
        s = '{name} ()'
        return s.format(name=self.__class__.__name__, **self.__dict__)
