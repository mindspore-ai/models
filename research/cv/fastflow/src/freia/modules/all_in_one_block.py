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

# This file was copied from code-completion [miguelvictor] [code-completion]
"""All in One Block Module."""

import warnings
from typing import Callable

import numpy as np
from scipy.stats import special_ortho_group

import mindspore
import mindspore.ops as ops
from mindspore import Parameter

from src.freia.modules.base import InvertibleModule

class AllInOneBlock(InvertibleModule):
    r"""Module combining the most common operations in a normalizing flow or similar model.

    It combines affine coupling, permutation, and global affine transformation
    ('ActNorm'). It can also be used as GIN coupling block, perform learned
    householder permutations, and use an inverted pre-permutation. The affine
    transformation includes a soft clamping mechanism, first used in Real-NVP.
    The block as a whole performs the following computation:
    .. math::
        y = V\\,R \\; \\Psi(s_\\mathrm{global}) \\odot \\mathrm{Coupling}\\Big(R^{-1} V^{-1} x\\Big)+ t_\\mathrm{global}
    - The inverse pre-permutation of x (i.e. :math:`R^{-1} V^{-1}`) is optional (see
      ``reverse_permutation`` below).
    - The learned householder reflection matrix
      :math:`V` is also optional all together (see ``learned_householder_permutation``
      below).
    - For the coupling, the input is split into :math:`x_1, x_2` along
      the channel dimension. Then the output of the coupling operation is the
      two halves :math:`u = \\mathrm{concat}(u_1, u_2)`.
      .. math::
          u_1 &= x_1 \\odot \\exp \\Big( \\alpha \\; \\mathrm{tanh}\\big( s(x_2) \\big)\\Big) + t(x_2) \\\\
          u_2 &= x_2
      Because :math:`\\mathrm{tanh}(s) \\in [-1, 1]`, this clamping mechanism prevents
      exploding values in the exponential. The hyperparameter :math:`\\alpha` can be adjusted.
    """

    def __init__(
            self,
            dims_in,
            dims_c=None,
            subnet_constructor: Callable = None,
            affine_clamping: float = 2.0,
            gin_block: bool = False,
            global_affine_init: float = 1.0,
            global_affine_type: str = "SOFTPLUS",
            permute_soft: bool = False,
            learned_householder_permutation: int = 0,
            reverse_permutation: bool = False,
        ):
        r"""Initialize.

        Args:
            dims_in (_type_): dims_in
            dims_c (list, optional): dims_c. Defaults to [].
            subnet_constructor (Callable, optional):
                class or callable ``f``, called as ``f(channels_in, channels_out)`` and should return a torch.nn.Module.
                Predicts coupling coefficients :math:`s, t`. Defaults to None.
            affine_clamping (float, optional):
                clamp the output of the multiplicative coefficients before exponentiation
                to +/- ``affine_clamping`` (see :math:`\\alpha` above). Defaults to 2.0.
            gin_block (bool, optional):
                Turn the block into a GIN block from Sorrenson et al, 2019.
                Makes it so that the coupling operations as a whole is volume preserving. Defaults to False.
            global_affine_init (float, optional):
                Initial value for the global affine scaling :math:`s_\mathrm{global}`.. Defaults to 1.0.
            global_affine_type (str, optional):
                ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. Defines the activation to be used
                on the beta for the global affine scaling (:math:`\\Psi` above).. Defaults to "SOFTPLUS".
            permute_soft (bool, optional):
                bool, whether to sample the permutation matrix :math:`R` from :math:`SO(N)`,
                or to use hard permutations instead. Note, ``permute_soft=True`` is very slow
                when working with >512 dimensions. Defaults to False.
            learned_householder_permutation (int, optional):
                Int, if >0, turn on the matrix :math:`V` above, that represents
                multiple learned householder reflections. Slow if large number.
                Dubious whether it actually helps network performance. Defaults to 0.
            reverse_permutation (bool, optional):
                Reverse the permutation before the block, as introduced by Putzky
                et al, 2019. Turns on the :math:`R^{-1} V^{-1}` pre-multiplication above. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """

        super(AllInOneBlock, self).__init__(dims_in, dims_c)

        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if not dims_c:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(
                dims_in[0][1:]
            ), f"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        self.permute_function = ops.Conv2D(channels, kernel_size=1)

        self.in_channels = channels
        self.clamp = affine_clamping
        self.GIN = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder = learned_householder_permutation

        if permute_soft and channels > 512:
            warnings.warn(
                (
                    "Soft permutation will take a very long time to initialize "
                    f"with {channels} feature channels. Consider using hard permutation instead."
                )
            )

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        self.global_affine_type = global_affine_type
        if global_affine_type == "SOFTPLUS":
            global_scale = 2.0 * np.log(np.exp(0.5 * 10.0 * global_affine_init) - 1)

        self.global_scale = Parameter(
            np.ones((1, self.in_channels, *([1] * self.input_rank)), dtype=np.float32) * float(global_scale)
        )
        self.global_scale_dim = tuple(range(len(self.global_scale.shape)))
        self.global_offset = Parameter(np.zeros((1, self.in_channels, *([1] * self.input_rank)), dtype=np.float32))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, j] = 1.0
        w = w.astype(np.float32)
        if self.householder:
            # instead of just the permutation matrix w, the learned housholder
            # permutation keeps track of reflection vectors vk, in addition to a
            # random initial permutation w_0.
            self.vk_householder = Parameter(0.2 * np.random.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = Parameter(w, requires_grad=False)
        else:
            self.w_perm = Parameter(
                w.reshape(channels, channels, *([1] * self.input_rank)), requires_grad=False)
            self.w_perm_inv = Parameter(
                (w.T).reshape(channels, channels, *([1] * self.input_rank)), requires_grad=False)

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor" "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None

        self.log = ops.Log()
        self.exp = ops.Exp()
        self.mm = ops.MatMul()
        self.eye = ops.Eye()
        self.tanh = ops.Tanh()
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.softplus = ops.Softplus()
        self.concat = ops.Concat(axis=1)
        self.sum = ops.ReduceSum(keep_dims=False)
        self.mean = ops.ReduceMean(keep_dims=False)
        self.mean_keep_dims = ops.ReduceMean(keep_dims=True)
        self.split = ops.Split(axis=1, output_num=len(self.splits))

    def _construct_householder_permutation(self):
        """Compute a permutation matrix.

        Compute a permutation matrix from the reflection vectors that are
        learned internally as nn.Parameters.
        """
        w = self.w_0
        for vk in self.vk_householder:
            w = self.mm(w, self.eye(self.in_channels) - 2 * ops.ger(vk, vk) / ops.dot(vk, vk))

        for _ in range(self.input_rank):
            w = self.expand_dims(w, -1)
        return w

    def _permute(self, x, rev=False):
        """Perform permutation.

        Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.
        """
        scale = 1.0
        perm_log_jac = 0.0
        if not self.GIN:
            if self.global_affine_type == "SOFTPLUS":
                beta = 0.5
                scale = 0.1 * (self.softplus(self.global_scale * beta) / beta)
                perm_log_jac = self.sum(self.log(scale), self.global_scale_dim)
            else:
                perm_log_jac = self.log(scale)

        output_result = None
        if rev:
            output_result = ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale, perm_log_jac)
        else:
            output_result = (self.permute_function(x * scale + self.global_offset, self.w_perm), perm_log_jac)
        return output_result

    def _pre_permute(self, x, rev=False):
        """Permute before the coupling block, only used if reverse_permutation is set."""

        output_result = None
        if rev:
            output_result = self.permute_function(x, self.w_perm)
        else:
            output_result = self.permute_function(x, self.w_perm_inv)

        return output_result

    def _affine(self, x, a, rev=False):
        """Perform affine coupling operation.

        Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet.
        """

        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a *= 0.1
        ch = x.shape[1]

        sub_jac = self.clamp * self.tanh(a[:, :ch])
        if self.GIN:
            sub_jac -= self.mean_keep_dims(sub_jac, self.sum_dims)

        output_result = None
        if not rev:
            output_result = (x * self.exp(sub_jac) + a[:, ch:], self.sum(sub_jac, self.sum_dims))
        else:
            output_result = ((x - a[:, ch:]) * ops.exp(-sub_jac), -self.sum(sub_jac, self.sum_dims))
        return output_result

    def construct(self, x, c=None, rev=False, jac=True):
        """See base class docstring."""

        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                input_perm = (1, 0, 2, 3)
                self.w_perm_inv = self.transpose(self.w_perm, input_perm)

        global_scaling_jac = 0
        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)

        x1, x2 = self.split(x[0])

        if self.conditional:
            c = list()
            c.append(x1)
            x1c = self.concat(c)
        else:
            x1c = x1

        if not rev:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = self.concat((x1, x2))

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        ch, h, w = x_out[0, :1].shape
        n_pixels = ch * h * w
        log_jac_det += (-1) ** self.cast(rev, mindspore.float32) * n_pixels * global_scaling_jac

        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        """Output Dims."""
        return input_dims
