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
"""poincare file"""
import mindspore.numpy as mnp
from mindspore.nn import Cell, Norm
from mindspore.ops import Shape, ReduceSum, Sqrt, ExpandDims, Tanh, Transpose, matmul, Pow, Reshape, clip_by_value
import mindspore.common.dtype as mstype
from src.math_utils import Artanh



class LorentzFactors(Cell):
    """lorentz_factors class"""

    def __init__(self, min_norm):
        """init"""
        super(LorentzFactors, self).__init__()
        self.min_norm = min_norm
        self.norm = Norm(axis=-1)

    def construct(self, x):
        """class construction"""
        x_norm = self.norm(x)
        return 1.0 / (1.0 - x_norm ** 2 + self.min_norm)


class ClampMin(Cell):
    """clamp_min class"""

    def __init__(self):
        """init fun"""
        super(ClampMin, self).__init__()
        self.shape = Shape()

    def construct(self, tensor, min1):
        """class construction"""
        min_mask = (tensor <= min1)
        min_mask1 = (tensor >= min1)
        min_add = mnp.ones(self.shape(tensor)) * min1 * min_mask
        return tensor * min_mask1 + min_add


class Proj(Cell):
    """proj class"""

    def __init__(self, min_norm):
        """init fun"""
        super(Proj, self).__init__()
        self.clamp_min = ClampMin()
        self.min_norm = min_norm
        self.norm_k = Norm(axis=-1, keep_dims=True)
        self.maxnorm = 1 - 4e-3

    def construct(self, x, c):
        """class construction"""
        norm = self.clamp_min(self.norm_k(x), self.min_norm)
        maxnorm = self.maxnorm / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return mnp.where(cond, projected, x)


class Clamp(Cell):
    """clamp class"""

    def __init__(self):
        super(Clamp, self).__init__()
        self.shape = Shape()

    def construct(self, tensor, min1, max1):
        """class construction"""
        return clip_by_value(tensor, min1, max1)


class Logmap0(Cell):
    """logmap0 class"""

    def __init__(self, min_norm):
        """init fun"""
        super(Logmap0, self).__init__()
        self.min_norm = min_norm
        self.norm_k = Norm(axis=-1, keep_dims=True)
        self.artanh = Artanh()
        self.norm_k = Norm(axis=-1, keep_dims=True)
        self.clamp_min = ClampMin()

    def construct(self, p, c):
        """class construction"""
        sqrt_c = c ** 0.5
        p_norm = self.clamp_min(self.norm_k(p), self.min_norm)
        scale = 1. / sqrt_c * self.artanh(sqrt_c * p_norm) / p_norm
        return scale * p


class KleinToPoincare(Cell):
    """klein to poincare class"""

    def __init__(self, min_norm):
        """init"""
        super(KleinToPoincare, self).__init__()
        self.min_norm = min_norm
        self.sqrt = Sqrt()
        self.sum = ReduceSum(keep_dims=True)
        self.proj = Proj(self.min_norm)

    def construct(self, x, c):
        """class construction"""
        x_poincare = x / (1.0 + self.sqrt(1.0 - self.sum(x * x, -1)))
        x_poincare = self.proj(x_poincare, c)
        return x_poincare


class ToKlein(Cell):
    """to klein class"""

    def __init__(self, min_norm):
        """init fun"""
        super(ToKlein, self).__init__()
        self.min_norm = min_norm
        self.sum = ReduceSum(keep_dims=True)
        self.klein_constraint = KleinConstraint(self.min_norm)

    def construct(self, x, c):
        """class construction"""
        x_2 = self.sum(x * x, -1)
        x_klein = 2 * x / (1.0 + x_2)
        x_klein = self.klein_constraint(x_klein)
        return x_klein


class KleinConstraint(Cell):
    """klein constraint class"""

    def __init__(self, min_norm):
        """init fun"""
        super(KleinConstraint, self).__init__()
        self.norm = Norm(axis=-1)
        self.min_norm = min_norm
        self.maxnorm = 1 - 4e-3
        self.shape = Shape()
        self.reshape = Reshape()

    def construct(self, x):
        """class construction"""
        last_dim_val = self.shape(x)[-1]
        norm = self.reshape(self.norm(x), (-1, 1))
        maxnorm = self.maxnorm
        cond = norm > maxnorm
        x_reshape = self.reshape(x, (-1, last_dim_val))
        projected = x_reshape / (norm + self.min_norm) * maxnorm
        x_reshape = mnp.where(cond, projected, x_reshape)
        x = self.reshape(x_reshape, self.shape(x))
        return x


class EinsteinMidpoint(Cell):
    """einstein mindpoint class"""

    def __init__(self, min_norm):
        """init fun"""
        super(EinsteinMidpoint, self).__init__()
        self.to_klein = ToKlein(min_norm)
        self.lorentz_factors = LorentzFactors(min_norm)
        self.sum = ReduceSum(keep_dims=True)
        self.unsqueeze = ExpandDims()
        self.sumFalse = ReduceSum(keep_dims=False)
        self.klein_constraint = KleinConstraint(min_norm)
        self.klein_to_poincare = KleinToPoincare(min_norm)

    def construct(self, x, c):
        """class construction"""
        x = self.to_klein(x, c)
        x_lorentz = self.lorentz_factors(x)
        x_norm = mnp.norm(x, axis=-1)
        # deal with pad value
        x_lorentz = (1.0 - (x_norm == 0.0).astype(mstype.float32)) * x_lorentz
        x_lorentz_sum = self.sum(x_lorentz, -1)
        x_lorentz_expand = self.unsqueeze(x_lorentz, -1)
        x_midpoint = self.sumFalse(x_lorentz_expand * x, 1) / x_lorentz_sum
        x_midpoint = self.klein_constraint(x_midpoint)
        x_p = self.klein_to_poincare(x_midpoint, c)
        return x_p


class ClampTanh(Cell):
    """clamp tanh class"""

    def __init__(self):
        """init fun"""
        super(ClampTanh, self).__init__()
        self.clamp = Clamp()
        self.tanh = Tanh()

    def construct(self, x, c=15):
        """class construction"""
        return self.tanh(self.clamp(x, -c, c))


class MobiusMatvec(Cell):
    """mobius matvec class"""

    def __init__(self, min_norm):
        """init fun"""
        super(MobiusMatvec, self).__init__()
        self.min_norm = min_norm
        self.norm_k = Norm(axis=-1, keep_dims=True)
        self.artanh = Artanh()
        self.norm_k = Norm(axis=-1, keep_dims=True)
        self.clamp_min = ClampMin()
        self.transpose = Transpose()
        self.clamp_tanh = ClampTanh()

    def construct(self, m, x, c):
        """class construction"""
        sqrt_c = c ** 0.5
        x_norm = self.clamp_min(self.norm_k(x), self.min_norm)
        mx = matmul(x, self.transpose(m, (1, 0)))
        mx_norm = self.clamp_min(self.norm_k(x), self.min_norm)
        t1 = self.artanh(sqrt_c * x_norm)
        t2 = self.clamp_tanh(mx_norm / x_norm * t1)
        res_c = t2 * mx / (mx_norm * sqrt_c)
        cond = mnp.array([[0]] * len(mx))
        res_0 = mnp.zeros(1)
        res = mnp.where(cond, res_0, res_c)
        return res


class Expmap0(Cell):
    """expmap0 class"""

    def __init__(self, min_norm):
        """init fun"""
        super(Expmap0, self).__init__()
        self.clamp_min = ClampMin()
        self.min_norm = min_norm
        self.clamp_tanh = ClampTanh()
        self.norm_k = Norm(axis=-1, keep_dims=True)

    def construct(self, u, c):
        """constructfun"""
        sqrt_c = c ** 0.5
        u_norm = self.clamp_min(self.norm_k(u), self.min_norm)
        gamma_1 = self.clamp_tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1


class MobiusAdd(Cell):
    """mobius add"""

    def __init__(self, min_norm):
        """init fun"""
        super(MobiusAdd, self).__init__()
        self.pow = Pow()
        self.sum = ReduceSum(keep_dims=True)
        self.clamp_min = ClampMin()
        self.min_norm = min_norm

    def construct(self, x, y, c, dim=-1):
        """constructfun"""
        x2 = self.sum(self.pow(x, 2), dim)
        y2 = self.sum(self.pow(y, 2), dim)
        xy = self.sum(x * y, dim)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / self.clamp_min(denom, self.min_norm)
