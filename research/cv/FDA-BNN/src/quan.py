# Copyright 2021 Huawei Technologies Co., Ltd
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
"""quantization"""
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import Parameter

class Signer(nn.Cell):
    '''
    take a real value x
    output sign(x)
    '''
    def __init__(self):
        super(Signer, self).__init__()
        self.sign = P.Sign()

    def construct(self, inp):
        """ construct """
        inp = inp * 1
        return self.sign(inp)

    def bprop(self, inp, out, dout):
        """ bprop """
        inp = inp * 1
        out = out * 1
        return (dout,)


def sign(inp):
    """ return sign(x) """
    return Signer.apply(inp)


class ScaleSigner(nn.Cell):
    '''
    take a real value x
    output sign(x) * mean(abs(x))
    '''
    def __init__(self):
        super(ScaleSigner, self).__init__()
        self.sign = P.Sign()
        self.mean = P.ReduceMean()
        self.abs = P.Abs()

    def construct(self, inp):
        """ construct """
        return self.sign(inp) * self.mean(self.abs(inp))

    def bprop(self, inp, out, dout):
        """ bprop """
        inp = inp * 1
        out = out * 1
        return (dout,)


class WQuan(nn.Cell):
    '''
    take a real value x
    output quantizer(x)
    '''
    def __init__(self):
        super(WQuan, self).__init__()
        self.quantizer = ScaleSigner()

    def construct(self, inp):
        """ construct """
        w = self.quantizer(inp)
        return w


class AQuan(nn.Cell):
    '''
    take a real value x
    output sign(x)
    '''
    def __init__(self):
        super(AQuan, self).__init__()
        self.quantizer = Signer()

    def construct(self, inp):
        """ construct """
        return self.quantizer(inp)


class QuanConv(nn.Cell):
    """ general QuanConv for quantized conv """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(QuanConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(Tensor(np.ones((out_channels, in_channels, kernel_size, kernel_size)), \
                                mindspore.float32), name="weight", requires_grad=True)
        self.quan_w = WQuan()
        self.quan_a = AQuan()
        self.conv2d = P.Conv2D(out_channel=out_channels,
                               kernel_size=kernel_size,
                               mode=1,
                               pad_mode="pad",
                               pad=padding,
                               stride=stride,
                               dilation=1,
                               group=1)

    def construct(self, inp):
        """ construct """
        w = self.quan_w(self.weight)
        x = self.quan_a(inp)
        output = self.conv2d(x, w)
        return output
