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
"""
Coupling layers.
"""
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype

import mindspore.nn as nn
from mindspore.ops import operations as ops
from mindspore.ops import functional as F
import mindspore.numpy as mnp

from .networks import NN
from .roundquant import RoundQuant


FORWARD_REMAINDER = [0, None]

_stack_axis1 = ops.Stack(axis=1)


def _do_integer_forward(z, frac_s, remainder):
    """MAT forward sub-process"""
    _, c = frac_s.shape
    tmp = mnp.concatenate([frac_s[:, -1:], frac_s[:, :-1]], 1)
    res = z * tmp

    lres = []

    for i in range(c):
        num = res[:, i] + remainder
        _r = F.tensor_floordiv(num, frac_s[:, i])
        remainder = num - _r * frac_s[:, i]
        lres.append(_r)

    res = _stack_axis1(lres)

    return res, remainder


def _do_integer_reverse(z, frac_s, remainder):
    """MAT inverse sub-process"""
    _, c = frac_s.shape
    res = z * frac_s

    lres = []

    for i in range(c - 1, -1, -1):
        num = res[:, i] + remainder
        _r = F.tensor_floordiv(num, frac_s[:, i - 1])
        remainder = num - _r * frac_s[:, i - 1]
        lres.append(_r)

    res = _stack_axis1(lres[::-1])

    return res, remainder


class SplitFactorCoupling(nn.Cell):
    """
    IDF Coupling layer.
    """
    def __init__(self, c_in, factor, args):
        super(SplitFactorCoupling, self).__init__()
        self.n_channels = args.n_channels
        self.kernel = 3
        self.input_channel = c_in

        if args.variable_type == 'discrete':
            self.round = RoundQuant(inverse_bin_width=2**args.n_bits)
            self.set_grad(False)
        else:
            self.round = None

        self.split_idx = c_in - (c_in // factor)

        self.nn = NN(
            n_channels=args.n_channels,
            c_in=self.split_idx,
            c_out=c_in - self.split_idx,
            depth=args.densenet_depth,
            kernel=self.kernel
        )

        self.alpha = Parameter(Tensor(.1, mstype.float32), requires_grad=True)

    def construct(self, z, ldj, reverse=False):
        """construct"""
        z1 = z[:, :self.split_idx, :, :]
        z2 = z[:, self.split_idx:, :, :]

        t = self.alpha * self.nn(z1)

        if self.round is not None:
            t = self.round(t)

        if not reverse:
            z2 = z2 + t
        else:
            z2 = z2 - t

        z = mnp.concatenate([z1, z2], axis=1)

        return z, ldj


class AffineFactorCoupling(nn.Cell):
    """
    VPF coupling layer.
    """
    def __init__(self, c_in, factor, args):
        super(AffineFactorCoupling, self).__init__()
        self.n_channels = args.n_channels
        self.kernel = 3
        self.input_channel = c_in
        self.first_numerator = float(2**16)

        if args.variable_type == 'discrete':
            self.round = RoundQuant(inverse_bin_width=2**args.n_bits)
            self.set_grad(False)
        else:
            self.round = None

        self.split_idx = c_in - (c_in // factor)
        self.factor_channel = c_in - self.split_idx

        self.nn = NN(
            n_channels=args.n_channels,
            c_in=self.split_idx,
            c_out=2 * (c_in - self.split_idx),
            depth=args.densenet_depth,
            kernel=self.kernel
        )

        self.alpha = Parameter(Tensor(1., mstype.float32), requires_grad=True)

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.tanh = ops.Tanh()
        self.exp = ops.Exp()
        self.cumprod = ops.CumProd()
        self.cast = ops.Cast()

    def _integer_forward(self, z, frac_s):
        """MAT forward algorithm"""
        global FORWARD_REMAINDER
        n, _ = frac_s.shape

        stage, remainder = FORWARD_REMAINDER
        if stage == 0:
            assert remainder is None
            remainder = mnp.zeros((n,), frac_s.dtype)
        else:
            assert len(remainder) == n

        res, remainder = _do_integer_forward(z, frac_s, remainder)

        FORWARD_REMAINDER[0] += 1
        FORWARD_REMAINDER[1] = remainder

        return res

    def _integer_reverse(self, z, frac_s):
        """MAT inverse algorithm"""
        global FORWARD_REMAINDER
        n, _ = frac_s.shape

        stage, remainder = FORWARD_REMAINDER
        assert stage > 0 and len(remainder) == n

        res, remainder = _do_integer_reverse(z, frac_s, remainder)

        FORWARD_REMAINDER[0] -= 1
        FORWARD_REMAINDER[1] = remainder
        if FORWARD_REMAINDER[0] == 0:
            # print (remainder)
            assert Tensor.all(remainder == 0)
            FORWARD_REMAINDER[1] = None

        return res

    def construct(self, z, ldj, reverse=False):
        """construct"""
        z1 = z[:, :self.split_idx, :, :]
        z2 = z[:, self.split_idx:, :, :]

        affine_params = self.alpha * self.nn(z1)
        t = affine_params[:, self.factor_channel:, :, :]
        logs = self.tanh(affine_params[:, :self.factor_channel, :, :])
        # make volume preserving
        s = self.exp(logs - mnp.mean(logs, axis=1, keepdims=True))

        if self.round is None:
            if not reverse:
                z2 = s * z2 + t
            else:
                z2 = (z2 - t) / s
        else:
            t = self.round(t)

            if reverse:
                z2 = z2 - t

            b, c, h, w = z2.shape
            s = self.transpose(s, (0, 2, 3, 1))
            s = self.reshape(s, (b, h * w * c))
            z2 = self.transpose(z2, (0, 2, 3, 1))
            z2 = self.reshape(z2, (b, h * w * c))
            # convert s to fractions
            cums = self.cumprod(s, 1)
            cums = cums / self.reshape(cums[:, -1], (-1, 1))
            frac_s = self.round.round(self.first_numerator / cums)
            frac_s[:, -1] = self.first_numerator
            frac_s = self.cast(frac_s, mstype.int64)

            # convert floating points to integer
            z2 = self.round.round(z2 * self.round.inverse_bin_width)
            z2 = self.cast(z2, mstype.int64)

            if not reverse:
                z2 = self._integer_forward(z2, frac_s)
            else:
                z2 = self._integer_reverse(z2, frac_s)

            # convert integer to floating points
            z2 = self.reshape(z2, (b, h, w, c))
            z2 = self.transpose(z2, (0, 3, 1, 2))
            z2 = self.cast(z2, mstype.float32)
            z2 = z2 / float(self.round.inverse_bin_width)

            if not reverse:
                z2 = z2 + t

        z = mnp.concatenate([z1, z2], axis=1)

        return z, ldj


class Coupling(nn.Cell):
    """
    Base coupling layer. Determine the type of
    coupling layer to use given args.
    """
    def __init__(self, c_in, args):
        super(Coupling, self).__init__()

        if args.split_quarter:
            factor = 4
        elif args.splitfactor > 1:
            factor = args.splitfactor
        else:
            factor = 2

        self.coupling = AffineFactorCoupling(
            c_in, factor, args=args)

    def construct(self, z, ldj, reverse=False):
        return self.coupling(z, ldj, reverse)
