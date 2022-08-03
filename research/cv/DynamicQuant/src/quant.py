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
"""Quantization"""

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore import Parameter

class Quantizer(nn.Cell):
    '''
    take a real value x
    output a discrete-valued x
    '''
    def __init__(self):
        super(Quantizer, self).__init__()
        self.round = ops.Rint()
        self.sum = ops.ReduceSum()
        self.scale = Parameter(Tensor(np.ones(1), mindspore.float32))

    def construct(self, inp, nbit, alpha=None, offset=None):
        self.scale = (2**nbit-1) if alpha is None else (2**nbit-1) / alpha
        if offset is None:
            out = self.round(inp * self.scale) / self.scale
        else:
            out = (self.round(inp * self.scale) + self.round(offset)) / self.scale
        return out

    def bprop(self, inp, out, dout):
        if self.offset is None:
            return dout, None, None, None
        return dout, None, None, self.sum(dout) / self.scale

class Signer(nn.Cell):
    '''
    take a real value x
    output sign(x)
    '''
    def __init__(self):
        super(Signer, self).__init__()
        self.sign = ops.Sign()

    def construct(self, inp):
        """ construct """
        inp = inp * 1
        return self.sign(inp)

    def bprop(self, inp, out, dout):
        """ bprop """
        inp = inp * 1
        out = out * 1
        return (dout,)

class ScaleSigner(nn.Cell):
    '''
    take a real value x
    output sign(x) * mean(abs(x))
    '''
    def __init__(self):
        super(ScaleSigner, self).__init__()
        self.sign = ops.Sign()
        self.mean = ops.ReduceMean()
        self.abs = ops.Abs()

    def construct(self, inp):
        """ construct """
        return self.sign(inp) * self.mean(self.abs(inp))

    def bprop(self, inp, out, dout):
        """ bprop """
        inp = inp * 1
        out = out * 1
        return (dout,)

class DoReFaW(nn.Cell):
    def __init__(self):
        super(DoReFaW, self).__init__()
        self.quantize = ScaleSigner()
        self.quantize2 = Quantizer()
        self.max = ops.ArgMaxWithValue()
        self.tanh = ops.Tanh()
        self.abs = ops.Abs()

    def construct(self, inp, nbit_w, *args, **kwargs):
        """ construct """
        if nbit_w == 1:
            w = self.quantize(inp)
        else:
            w = self.tanh(inp)
            maxv = self.abs(w)
            for _ in range(len(inp.shape)):
                _, maxv = self.max(maxv)
            w = w / (2 * maxv) + 0.5
            w = 2 * self.quantize2(w, nbit_w) - 1
        return w

class DoReFaA(nn.Cell):
    def __init__(self):
        super(DoReFaA, self).__init__()
        self.quantize = Quantizer()

    def construct(self, inp, nbit_a, *args, **kwargs):
        """ construct """
        a = ops.clip_by_value(inp, 0, 1)
        a = self.quantize(a, nbit_a, *args, **kwargs)
        return a

class PACTA(nn.Cell):
    def __init__(self):
        super(PACTA, self).__init__()
        self.quantize = Quantizer()
        self.abs = ops.Abs()

    def construct(self, inp, nbit_a, alpha, *args, **kwargs):
        """ construct """
        x = 0.5 * (self.abs(inp) - self.abs(inp-alpha) + alpha)
        return self.quantize(x, nbit_a, alpha, *args, **kwargs)

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


class QuanConv(nn.Conv2d):
    # general quantization for quantized conv
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w, quan_name_a, nbit_w=32, nbit_a=32,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, has_offset=False):
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, "pad", padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': DoReFaW, 'pact': DoReFaW}
        name_a_dict = {'dorefa': DoReFaA, 'pact': PACTA}
        self.quan_w = name_w_dict[quan_name_w]()
        self.quan_a = name_a_dict[quan_name_a]()

        if quan_name_a == 'pact':
            self.alpha_a = Parameter(Tensor(np.ones(1), mindspore.float32))
        else:
            self.alpha_a = None

        if quan_name_w == 'pact':
            self.alpha_w = Parameter(Tensor(np.ones(1), mindspore.float32))
        else:
            self.alpha_w = None

        if has_offset:
            self.offset = Parameter(Tensor(np.zeros(1), mindspore.float32))
        else:
            self.offset = None

        if bias:
            self.bias = Parameter(Tensor(np.zeros(out_channels), mindspore.float32))
        else:
            self.bias = None

        self.conv2d = ops.Conv2D(out_channel=out_channels,
                                 kernel_size=kernel_size,
                                 mode=1,
                                 stride=stride,
                                 pad_mode="pad",
                                 pad=padding,
                                 dilation=dilation,
                                 group=groups)

    def construct(self, inp):
        # w quan
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x = self.quan_a(inp, self.nbit_a, self.alpha_a)
        else:
            x = inp

        x = self.conv2d(x, w)
        return x

class DymQuanConv(nn.Conv2d):
    # dynamic quantization for quantized conv
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w, quan_name_a, nbit_w, nbit_a,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, has_offset=False):
        super(DymQuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, "pad", padding, dilation, groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': DoReFaW, 'pact': DoReFaW}
        name_a_dict = {'dorefa': DoReFaA, 'pact': PACTA}
        self.quan_w = name_w_dict[quan_name_w]()
        self.quan_a = name_a_dict[quan_name_a]()

        if quan_name_a == 'pact':
            self.alpha_a = Parameter(Tensor(np.ones(1), mindspore.float32))
        else:
            self.alpha_a = None

        if quan_name_w == 'pact':
            self.alpha_w = Parameter(Tensor(np.ones(1), mindspore.float32))
        else:
            self.alpha_w = None

        if has_offset:
            self.offset = Parameter(Tensor(np.zeros(1), mindspore.float32))
        else:
            self.offset = None

        if bias:
            self.bias = Parameter(Tensor(np.zeros(out_channels), mindspore.float32))
        else:
            self.bias = None

        self.conv2d = ops.Conv2D(out_channel=out_channels,
                                 kernel_size=kernel_size,
                                 mode=1,
                                 stride=stride,
                                 pad_mode="pad",
                                 pad=padding,
                                 dilation=dilation,
                                 group=groups)
        self.expand_dims = ops.ExpandDims()

    def construct(self, inp, mask):
        # w quan
        w0 = self.quan_w(self.weight, self.nbit_w-1, self.alpha_w, self.offset)
        w1 = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        w2 = self.quan_w(self.weight, self.nbit_w+1, self.alpha_w, self.offset)

        # a quan
        x0 = self.quan_a(inp, self.nbit_a-1, self.alpha_a)
        x1 = self.quan_a(inp, self.nbit_a, self.alpha_a)
        x2 = self.quan_a(inp, self.nbit_a+1, self.alpha_a)

        x0 = self.conv2d(x0, w0)
        x1 = self.conv2d(x1, w1)
        x2 = self.conv2d(x2, w2)

        x = x0*self.expand_dims(self.expand_dims(self.expand_dims(mask[:, 0], 1), 2), 3)+ \
            x1*self.expand_dims(self.expand_dims(self.expand_dims(mask[:, 1], 1), 2), 3)+ \
            x2*self.expand_dims(self.expand_dims(self.expand_dims(mask[:, 2], 1), 2), 3)

        return x
