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
The flow model.
"""
import numpy as np
from mindspore import dtype as mstype

import mindspore.nn as nn
from mindspore.ops import operations as ops
import mindspore.numpy as mnp

from .prior import Prior
from .generative_flow import GenerativeFlow
from .loss import compute_loss_array


class Normalize(nn.Cell):
    """Normalize layer."""
    def __init__(self, args):
        super(Normalize, self).__init__()
        self.n_bits = args.n_bits
        self.variable_type = args.variable_type
        self.input_size = args.input_size

    def construct(self, x, ldj, reverse=False):
        """construct"""
        domain = 2.**self.n_bits

        if self.variable_type == 'discrete':
            # Discrete variables will be measured on intervals sized 1/domain.
            # Hence, there is no need to change the log Jacobian determinant.
            dldj = 0
        elif self.variable_type == 'continuous':
            dldj = -np.log(domain) * np.prod(self.input_size)
        else:
            raise ValueError

        if not reverse:
            x = (x - domain / 2) / domain
            ldj += dldj
        else:
            x = x * domain + domain / 2
            ldj -= dldj

        return x, ldj


class Model(nn.Cell):
    """
    The flow model.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.variable_type = args.variable_type
        self.distribution_type = args.distribution_type

        n_channels, height, width = args.input_size

        self.normalize = Normalize(args)

        self.flow = GenerativeFlow(n_channels, height, width, args)

        self.n_bits = args.n_bits
        self.base_bits = 8

        self.z_size = self.flow.z_size

        self.prior = Prior(self.z_size, args)

        self.uniform = ops.UniformReal()
        self.cast = ops.Cast()
        self.zeros_like = ops.ZerosLike()
        self.floor = ops.Floor()

    def dequantize(self, x):
        """dequantize"""
        if self.training:
            x = x + self.uniform(x.shape)
        else:
            # Required for stability.
            alpha = 1e-3
            x = x + alpha + self.uniform(x.shape) * (1 - 2 * alpha)

        return x

    def loss(self, pz, z, pys, ys, ldj):
        """compute loss (log likelihood and bpd)"""
        loss, bpd, bpd_per_prior = \
            compute_loss_array(pz, z, pys, ys, ldj, self.args)

        return loss, bpd, bpd_per_prior

    def construct(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log
         det jacobian is zero for a plain VAE (without flows), and z_0 = z_k.
        """
        # Decode z to x.

        assert x.dtype == mstype.uint8

        x = self.cast(x, mstype.float32)

        ldj = mnp.zeros(x.shape[0])
        level = 2.**(self.n_bits - self.base_bits)
        if self.variable_type == 'continuous':
            x = self.dequantize(x)
            x = x * level
            ldj += np.log(level) * x[0].size
        elif self.variable_type == 'discrete':
            x = x * level + self.floor(level * self.uniform(x.shape))
            ldj += np.log(level) * x[0].size
        else:
            raise ValueError

        x, ldj = self.normalize(x, ldj)

        z, ldj, pys, ys = self.flow(x, ldj, pys=(), ys=())

        pz, z, ldj = self.prior(z, ldj)

        loss, bpd, bpd_per_prior = self.loss(pz, z, pys, ys, ldj)

        return loss, bpd, bpd_per_prior, pz, z, pys, ys, ldj

    def inverse(self, z, ys):
        """inverse operation of the flow model"""
        ldj = self.zeros_like(z[:, 0, 0, 0])
        x, ldj, _, _ = \
            self.flow(z, ldj, pys=[], ys=ys, reverse=True)

        x, ldj = self.normalize(x, ldj, reverse=True)

        level = 2.**(self.n_bits - self.base_bits)
        x = self.floor(x / level)

        x = ops.Minimum()(x, 255)
        x = ops.Maximum()(x, 0)

        x_uint8 = self.cast(x, mnp.uint8)

        return x_uint8
