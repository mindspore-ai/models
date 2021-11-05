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
Main body of flow model.
"""
import numpy as np
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype

import mindspore.nn as nn
from mindspore.ops import operations as ops

from .coupling import Coupling
from .lu import LUConv1x1
from .prior import SplitPrior


inner_layers = []


class Flatten(nn.Cell):
    """Flatten operation."""
    def construct(self, x):
        """construct"""
        return x.view(x.shape[0], -1)


class Permute(nn.Cell):
    """Channel permute operation."""
    def __init__(self, n_channels):
        super(Permute, self).__init__()

        permutation = np.arange(n_channels, dtype=np.int32)
        np.random.shuffle(permutation)

        permutation_inv = np.zeros(n_channels, dtype=np.int32)
        permutation_inv[permutation] = np.arange(n_channels, dtype=np.int32)

        self.permutation = Parameter(
            Tensor.from_numpy(permutation),
            requires_grad=False)
        self.permutation_inv = Parameter(
            Tensor.from_numpy(permutation_inv),
            requires_grad=False)

    def construct(self, z, ldj, reverse=False):
        """construct"""
        assert z.dtype == mstype.float32, 'Permute layer only support float32.'
        if not reverse:
            z = z[:, self.permutation, :, :]
        else:
            z = z[:, self.permutation_inv, :, :]

        return z, ldj

    def InversePermute(self):
        """get inverse of permute operation"""
        inv_permute = Permute(len(self.permutation))
        inv_permute.permutation = self.permutation_inv
        inv_permute.permutation_inv = self.permutation
        return inv_permute


class Squeeze(nn.Cell):
    """Squeeze operation."""
    def __init__(self):
        super(Squeeze, self).__init__()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, z, ldj, reverse=False):
        """construct"""
        if not reverse:
            zs = z.shape
            z = z.view(zs[0], zs[1], zs[2] // 2, 2, zs[3] // 2, 2)
            z = self.transpose(z, (0, 1, 3, 5, 2, 4))
            z = z.view(zs[0], zs[1] * 4, zs[2] // 2, zs[3] // 2)
        else:
            zs = z.shape
            z = z.view(zs[0], zs[1] // 4, 2, 2, zs[2], zs[3])
            z = self.transpose(z, (0, 1, 4, 2, 5, 3))
            z = z.view(zs[0], zs[1] // 4, zs[2] * 2, zs[3] * 2)
        return z, ldj


class GenerativeFlow(nn.Cell):
    """Main body of flow model."""
    def __init__(self, n_channels, height, width, args):
        super(GenerativeFlow, self).__init__()
        layers = []
        layers.append(Squeeze())
        n_channels *= 4
        height //= 2
        width //= 2

        for level in range(args.n_levels):

            for _ in range(args.n_flows):
                layers.append(LUConv1x1(n_channels, args))
                layers.append(Permute(n_channels))
                layers.append(Coupling(n_channels, args))

            if level < args.n_levels - 1:
                if args.splitprior:
                    # Standard splitprior
                    factor_out = n_channels // 2
                    layers.append(SplitPrior(n_channels, factor_out, args))
                    n_channels = n_channels - factor_out

                layers.append(Squeeze())
                n_channels *= 4
                height //= 2
                width //= 2

        self.layers = nn.CellList(layers)
        self.z_size = (n_channels, height, width)

    def construct(self, z, ldj, pys=(), ys=(), reverse=False):
        """construct"""
        if not reverse:
            for _, layer in enumerate(self.layers):
                if isinstance(layer, (SplitPrior)):
                    py, y, z, ldj = layer(z, ldj)
                    pys += (py,)
                    ys += (y,)
                else:
                    z, ldj = layer(z, ldj)
        else:
            for _, layer in reversed(list(enumerate(self.layers))):
                if isinstance(layer, (SplitPrior)):
                    if ys:
                        z, ldj = layer.inverse(z, ldj, y=ys[-1])
                        # Pop last element
                        ys = ys[:-1]
                    else:
                        z, ldj = layer.inverse(z, ldj, y=None)
                else:
                    z, ldj = layer(z, ldj, reverse=True)

        return z, ldj, pys, ys
