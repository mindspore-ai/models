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
Priors.
"""
import numpy as np
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype

import mindspore.nn as nn
from mindspore.ops import operations as ops
import mindspore.numpy as mnp

from .networks import NN


class Prior(nn.Cell):
    """The prior of flow model. Compute p(z)."""
    def __init__(self, size, args):
        super().__init__()
        c, h, w = size

        self.inverse_bin_width = 2**args.n_bits
        self.variable_type = args.variable_type
        self.distribution_type = args.distribution_type
        self.n_mixtures = args.n_mixtures

        if self.n_mixtures == 1:
            self.mu = Parameter(mnp.zeros((c, h, w), mstype.float32), requires_grad=True)
            self.logs = Parameter(mnp.zeros((c, h, w), mstype.float32), requires_grad=True)
        elif self.n_mixtures > 1:
            self.mu = Parameter(Tensor(np.ones((c, h, w, self.n_mixtures)) * np.linspace(-(self.n_mixtures - 1) / \
                                2, (self.n_mixtures - 1) / 2, self.n_mixtures), mstype.float32), requires_grad=True)
            self.logs = Parameter(mnp.zeros((c, h, w, self.n_mixtures), mstype.float32), requires_grad=True)
            self.pi_logit = Parameter(mnp.zeros((c, h, w, self.n_mixtures), mstype.float32), requires_grad=True)

        self.softmax = ops.Softmax(axis=-1)
        self.tile = ops.Tile()

    def get_pz(self, n):
        """get parameters of the prior"""
        if self.n_mixtures == 1:
            mu = self.tile(mnp.expand_dims(self.mu, 0), (n, 1, 1, 1))
            logs = self.tile(mnp.expand_dims(self.logs, 0), (n, 1, 1, 1))  # scaling scale
            return mu, logs

        # self.n_mixtures > 1:
        pi = self.softmax(self.pi_logit)
        mu = self.tile(mnp.expand_dims(self.mu, 0), (n, 1, 1, 1, 1))
        logs = self.tile(mnp.expand_dims(self.logs, 0), (n, 1, 1, 1, 1))
        pi = self.tile(mnp.expand_dims(pi, 0), (n, 1, 1, 1, 1))
        return mu, logs, pi

    def construct(self, z, ldj):
        """construct"""
        pz = self.get_pz(z.shape[0])

        return pz, z, ldj

    def sample(self, n):
        """sample data with prior"""
        raise NotImplementedError()

    def decode(self, states, n, decode_fn):
        """decode data with prior"""
        raise NotImplementedError()


class SplitPrior(nn.Cell):
    """Split prior layer: compute p(y|z)"""
    def __init__(self, c_in, factor_out, args):
        super().__init__()

        self.split_idx = c_in - factor_out
        self.inverse_bin_width = 2**args.n_bits
        self.variable_type = args.variable_type
        self.distribution_type = args.distribution_type
        self.input_channel = c_in
        self.kernel = 3

        self.nn = NN(
            n_channels=args.n_channels,
            c_in=c_in - factor_out,
            c_out=factor_out * 2,
            depth=args.densenet_depth,
            kernel=self.kernel
        )

        self.gamma = Parameter(Tensor(0.1, mstype.float32), requires_grad=True)
        self.delta = Parameter(Tensor(0.1, mstype.float32), requires_grad=True)

    def get_py(self, z):
        """get parameters of p(y|z)"""
        h = self.nn(z)
        mu = h[:, ::2, :, :]
        logs = h[:, 1::2, :, :]

        py = [self.gamma * mu, self.delta * logs]

        return py

    def split(self, z):
        """split"""
        z1 = z[:, :self.split_idx, :, :]
        y = z[:, self.split_idx:, :, :]
        return z1, y

    def combine(self, z, y):
        """combine"""
        result = mnp.concatenate([z, y], axis=1)

        return result

    def construct(self, z, ldj):
        """construct"""
        z, y = self.split(z)

        py = self.get_py(z)

        return py, y, z, ldj

    def inverse(self, z, ldj, y):
        """inverse operation"""
        # Sample if y is not given.
        if y is None:
            raise NotImplementedError()

        z = self.combine(z, y)

        return z, ldj

    def decode(self, z, ldj, states, decode_fn):
        """decode from p(y|z)"""
        raise NotImplementedError()
