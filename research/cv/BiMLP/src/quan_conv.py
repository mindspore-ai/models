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

import mindspore
import mindspore.nn as nn
from mindspore import ops


class Quantizer(nn.Cell):
    """
    take a real value x in alpha*[0,1] or alpha*[-1,1]
    output a discrete-valued x in alpha*{0, 1/(2^k-1), ..., (2^k-1)/(2^k-1)} or likeness
    where k is nbit
    """

    def __init__(self):
        super(Quantizer, self).__init__()
        self.round = ops.Rint()
        self.sum = ops.ReduceSum()

    def construct(self, inp, nbit, alpha=None, offset=None):
        scale = (2**nbit - 1) if alpha is None else (2 **
                                                     nbit - 1) / float(alpha)

        return (
            self.round(inp * scale) / scale
            if offset is None
            else (self.round(inp * scale) + self.round(offset)) / scale
        )


# standard sign with STE
class Signer(nn.Cell):
    """
    take a real value x
    output sign(x)
    """

    def __init__(self):
        super(Signer, self).__init__()
        self.sign = ops.Sign()

    def construct(self, inp):
        return self.sign(inp)

    def bprop(self, inp, out, dout):
        return (dout,)


# sign in xnor-net for weights
class XnorM(nn.Cell):
    """
    take a real value x
    output sign(x_c) * E(|x_c|)
    """

    def construct(self, inp):
        return ops.Sign()(inp) * ops.ReduceMean(keep_dims=True)(ops.Abs()(inp), 1)

    def bprop(self, inp, out, dout):
        return (dout,)


# sign in dorefa-net for weights
class ScaleSigner(nn.Cell):
    """
    take a real value x
    output sign(x) * E(|x|)
    """

    def __init__(self):
        super(ScaleSigner, self).__init__()
        self.sign = ops.Sign()
        self.mean = ops.ReduceMean()
        self.abs = ops.Abs()

    def construct(self, inp):
        return self.sign(inp) * self.mean(self.abs(inp))

    def bprop(self, inp, out, dout):
        return (dout,)


def binary_w(w, *args, **kwargs):
    return Signer()(ops.clip_by_value(w, clip_value_min=-1, clip_value_max=1))


def dorefa_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = ScaleSigner()(w)
    else:
        me_abs = ops.Abs()
        me_tanh = ops.Tanh()
        w = me_tanh(w)

        w = w / (2 * ops.ArgMaxWithValue()(me_abs(w))[1]) + 0.5
        w = 2 * Quantizer()(w, nbit_w) - 1
    return w


def wrpn_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = ScaleSigner()(w)
    else:
        w = Quantizer()(
            ops.clip_by_value(w, clip_value_min=-1,
                              clip_value_max=1), nbit_w - 1
        )
    return w


def xnor_w(w, nbit_w, *args, **kwargs):
    if nbit_w != 1:
        raise ValueError("nbit_w must be 1 in XNOR-Net.")
    return XnorM()(w)


def bireal_w(w, nbit_w, *args, **kwargs):
    me_abs = ops.Abs()
    me_mean = ops.ReduceMean()
    if nbit_w != 1:
        raise ValueError("nbit_w must be 1 in Bi-Real-Net.")
    return Signer()(w) * me_mean(me_abs(w.clone().detach()))


def binary_a(inp, *args, **kwargs):
    return Signer()(ops.clip_by_value(inp, clip_value_min=-1, clip_value_max=1))


# dorefa quantize for activations
def dorefa_a(inp, nbit_a, *args, **kwargs):
    return Quantizer()(
        ops.clip_by_value(inp, clip_value_min=-1, clip_value_max=1),
        nbit_a,
        *args,
        **kwargs
    )


# PACT quantize for activations
def pact_a(inp, nbit_a, alpha, *args, **kwargs):
    me_abs = ops.Abs()
    x = 0.5 * (me_abs(inp) - me_abs(inp - alpha) + alpha)
    return Quantizer()(x, nbit_a, alpha, *args, **kwargs)


# bi-real sign for activations
class BirealActivation(nn.Cell):
    """
    take a real value x
    output sign(x)
    """

    def __init__(self):
        super(BirealActivation, self).__init__()
        self.sign = ops.Sign()
        self.lt = ops.Less()
        self.ge = ops.GreaterEqual()
        self.cast = ops.Cast()

    def construct(self, inp, nbit_a=1):
        return self.sign(ops.clip_by_value(inp, clip_value_min=-1, clip_value_max=1))

    def bprop(self, inp, nbit_a, out, dout):
        grad_inp = (2 + 2 * inp) * self.cast(
            self.lt(inp, 0), mindspore.float32
        ) + (2 - 2 * inp) * self.cast(self.ge(inp, 0), mindspore.float32)
        grad_inp = ops.clip_by_value(
            grad_inp, clip_value_min=0, clip_value_max=1e26
        )
        grad_inp *= dout
        return (grad_inp, None)


def bireal_a(inp, nbit_a, *args, **kwargs):
    return BirealActivation()(inp)


# new sigmoid
class TSigmoid(nn.Cell):
    """
    construct: sigmoid(T*x)
    bprop: freezing T=1
    """

    def __init__(self):
        super(TSigmoid, self).__init__()
        self.sigmoid = ops.Sigmoid()

    def construct(self, inp, T):
        return self.sigmoid(T * inp)

    def bprop(self, inp, T, out, dout):
        x = self.sigmoid(inp)
        grad_inp = x * (1 - x)
        grad_inp *= dout
        return (grad_inp, None)


def t_sigmoid(inp, T, *args, **kwargs):
    sigmoid = ops.Sigmoid()
    return sigmoid(inp * T)


def quan_net_w(inp, alpha, beta, bias, T, training):
    if training:
        return 2 * alpha * t_sigmoid(beta * inp - bias, T) - 1
    me_gt = ops.Greater()
    me_cast = ops.Cast()
    return 2 * alpha * me_cast(me_gt(beta * inp - bias, 0), mindspore.float32) - 1


def quan_net_a(inp, alpha, beta, bias, T, training):
    if training:
        return alpha * (t_sigmoid(beta * inp - bias, T))
    me_gt = ops.Greater()
    me_cast = ops.Cast()
    return alpha * (me_cast(me_gt(beta * inp - bias, 0), mindspore.float32))


def step(x, b):
    """The step function in test stage."""
    y = ops.ZerosLike()(x)
    me_gt = ops.Greater()
    mask = me_gt(x - b, 0.0)
    y[mask] = 1.0
    return y


def laplace_boundary(p, m, b):
    me_abs = ops.Abs()
    me_sign = ops.Sign()
    me_log = ops.Log()
    return m - b * me_sign(p - 0.5) * me_log(1 - 2 * me_abs(p - 0.5))


class QuanConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, pad_mode='pad', quan_name_w='dorefa',
                 quan_name_a='dorefa', nbit_w=1, nbit_a=1, has_offset=False,
                 stride=1, padding=0, dilation=1, group=1, bias=False):
        pad_mode = "pad"
        if isinstance(padding, tuple) and len(padding) == 2:
            print("wrong padding")
        super(QuanConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_offset,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.quan_name_w = quan_name_w
        self.quan_name_a = quan_name_a

        name_w_dict = {
            "bnn": binary_w,
            "dorefa": dorefa_w,
            "pact": dorefa_w,
            "wrpn": wrpn_w,
            "xnor": xnor_w,
            "bireal": bireal_w,
        }
        name_a_dict = {
            "bnn": binary_a,
            "dorefa": dorefa_a,
            "pact": pact_a,
            "wrpn": dorefa_a,
            "xnor": dorefa_a,
            "bireal": bireal_a,
        }
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]

        diff_channels = self.out_channels - self.in_channels
        self.pad = nn.Pad(
            paddings=(
                (0, 0),
                (diff_channels // 2, diff_channels - diff_channels // 2),
                (0, 0),
                (0, 0),
            ),
            mode="CONSTANT",
        )

        self.relu = ops.ReLU()

    def construct(self, inp):
        if self.nbit_w == 0 or self.nbit_a == 0:

            if self.stride == 2 or self.stride == (2, 2):
                x = self.pad(inp[:, :, ::2, ::2])
                return x
            x = self.pad(inp)
            return x

        # w quan
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x = self.quan_a(inp, self.nbit_a)
        else:
            x = self.relu(inp)
        x = self.conv2d(x, w)
        return x
