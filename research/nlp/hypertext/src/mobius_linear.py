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
"""mobius liner"""
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.nn import Cell
from mindspore.ops import Zeros
from numpy import ones
from numpy.random import randn
from src.poincare import Proj, MobiusMatvec, Expmap0, MobiusAdd


class MobiusLinear(Cell):
    """Mobius linear layer."""

    def __init__(self, in_features, out_features, c, use_bias=True):
        """init fun"""
        super(MobiusLinear, self).__init__()
        self.zeros = Zeros()
        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.bias = Parameter(Tensor(ones([1, out_features]), mstype.float32))
        self.weight = Parameter(
            Tensor(randn(out_features, in_features), mstype.float32))

        self.min_norm = 1e-15
        self.mobius_matvec = MobiusMatvec(self.min_norm)
        self.proj = Proj(self.min_norm)
        self.expmap0 = Expmap0(min_norm=self.min_norm)
        self.mobius_add = MobiusAdd(self.min_norm)

    def construct(self, x):
        """class construction"""
        mv = self.mobius_matvec(self.weight, x, self.c)
        res = self.proj(mv, self.c)
        if self.use_bias:
            proj_tan0 = self.bias.view(1, -1)
            bias = proj_tan0
            hyp_bias = self.expmap0(bias, self.c)
            hyp_bias = self.proj(hyp_bias, self.c)
            res = self.mobius_add(res, hyp_bias, c=self.c)
            res = self.proj(res, self.c)
        return res
