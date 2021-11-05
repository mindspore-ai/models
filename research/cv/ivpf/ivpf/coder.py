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
Coder with flow model.
"""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

import mindspore.numpy as mnp

import fast_ans
from .coupling import FORWARD_REMAINDER
from .prior import SplitPrior


class Coder:
    """
    Coder with flow model.
    """
    def __init__(self, model):
        self.args = model.args
        self.n_coupling = model.args.n_levels * model.args.n_flows

        self.model = model
        self._disc_z = fast_ans.Discretization(-8, 8, model.n_bits)
        self._disc_u = fast_ans.Discretization(
            0, 1, model.n_bits - model.base_bits)
        self._ans = fast_ans.ANS(60, 2 ** 20, 0, 16)

    def ans_length(self):
        """the stream length of rANS coder"""
        return self._ans.stream_length()

    def get_forward_remainders(self):
        """store the remainder for encoding"""
        stage, remainder = FORWARD_REMAINDER
        FORWARD_REMAINDER[0] = 0
        FORWARD_REMAINDER[1] = None
        return [stage, remainder]

    def set_forward_remainders(self, remainders):
        """load the remainder for decoding"""
        assert FORWARD_REMAINDER[0] == 0 and FORWARD_REMAINDER[1] is None
        FORWARD_REMAINDER[0] = remainders[0]
        FORWARD_REMAINDER[1] = remainders[1]

    def encode(self, x):
        """encoding process"""
        assert x.dtype == mstype.uint8
        assert self.model.n_bits >= self.model.base_bits
        assert self.model.distribution_type == 'normal'

        x = self.model.cast(x, mstype.float32)

        ldj = mnp.zeros(x.shape[0])

        # dequantization by decode
        if self.model.n_bits > self.model.base_bits:
            level = 2. ** (self.model.n_bits - self.model.base_bits)

            sym_u = self._ans.decode_uniform_diag(self._disc_u, x.size)
            u = Tensor(self._disc_u.symbol_to_real(sym_u), mstype.float32)
            u = self.model.floor(level * u) / level
            u = u.view(*x.shape)

            x = level * x + self.model.floor(level * u)
            ldj += np.log(level) * x[0].size

        x, ldj = self.model.normalize(x, ldj)
        z, ldj, pys, ys = self.model.flow(x, ldj, pys=(), ys=())
        pz, z, ldj = self.model.prior(z, ldj)

        pjs = list(pys) + [pz]
        js = list(ys) + [z]

        for pj, j in zip(pjs, js):
            sym_j = self._disc_z.real_to_symbol(
                j.asnumpy().astype(np.float64).ravel())
            if len(pj) == 2:
                self._ans.encode_gaussian_diag(
                    sym_j,
                    pj[0].asnumpy().astype(np.float64).ravel(),
                    np.exp(0.5 * pj[1].asnumpy().astype(np.float64)).ravel(),
                    self._disc_z,
                    True
                )
            else:
                assert len(pj) == 3
                n_mixtures = pj[0].shape[-1]
                self._ans.encode_mixed_gaussian_diag(
                    sym_j,
                    pj[0].asnumpy().astype(np.float64).reshape((-1, n_mixtures)),
                    np.exp(0.5 * pj[1].asnumpy().astype(np.float64).reshape((-1, n_mixtures))),
                    pj[2].asnumpy().astype(np.float64).reshape((-1, n_mixtures)),
                    self._disc_z,
                    True
                )

        fwd_remainder = self.get_forward_remainders()
        return self._ans, fwd_remainder

    def decode_prior(self, n):
        """decoding sub-process: decoding prior"""
        pz = self.model.prior.get_pz(n)
        n_mixtures = pz[0].shape[-1]

        sym_z = self._ans.decode_mixed_gaussian_diag(
            pz[0].asnumpy().astype(np.float64).reshape((-1, n_mixtures)),
            np.exp(0.5 * pz[1].asnumpy().astype(np.float64).reshape((-1, n_mixtures))),
            pz[2].asnumpy().astype(np.float64).reshape((-1, n_mixtures)),
            self._disc_z,
            True
        )

        z = self._disc_z.symbol_to_real(sym_z).reshape(pz[0].shape[:4])
        z = Tensor(z, mstype.float32)
        return z

    def decode_split_prior(self, cell, z, ldj):
        """decoding sub-process: decoding split prior"""
        py = cell.get_py(z)

        sym_y = self._ans.decode_gaussian_diag(
            py[0].asnumpy().astype(np.float64).ravel(),
            np.exp(0.5 * py[1].asnumpy().astype(np.float64)).ravel(),
            self._disc_z,
            True
        )

        y = self._disc_z.symbol_to_real(sym_y).reshape(py[0].shape[:4])
        y = Tensor(y, mstype.float32)

        return cell.combine(z, y), ldj

    def decode(self, batchsize, fwd_remainder):
        """decoding process"""
        assert self.model.distribution_type == 'normal'
        self.set_forward_remainders(fwd_remainder)

        z = self.decode_prior(batchsize)

        ldj = mnp.zeros(z.shape[0])

        for _, layer in reversed(list(enumerate(self.model.flow.layers))):
            if isinstance(layer, SplitPrior):
                z, ldj = self.decode_split_prior(layer, z, ldj)
            else:
                z, ldj = layer(z, ldj, reverse=True)

        x, ldj = self.model.normalize(z, ldj, reverse=True)

        # recover x and encode dequantizer
        if self.model.n_bits > self.model.base_bits:
            level = 2. ** (self.model.n_bits - self.model.base_bits)
            xu = x / level
            x = self.model.floor(xu)
            u = xu - x

            sym_u = self._disc_u.real_to_symbol(
                u.asnumpy().astype(np.float64)).ravel()
            self._ans.encode_uniform_diag(sym_u, self._disc_u)

        x = self.model.cast(x, mstype.uint8)

        return self._ans, x
