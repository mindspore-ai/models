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
######################## conv1d with weight normalization  ########################
"""
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Tensor, Parameter


def norm_except_dim(v, pow1, dim):
    """norm_except_dim """
    if dim == -1:
        result = mnp.norm(v, pow1)
    elif dim == 0:
        output_size = (v.shape[0],) + (1,) * (v.ndim - 1)
        result = mnp.norm(v.view((v.shape[0], -1)), pow1, 1).view(output_size)
    elif dim == (v.ndim - 1):
        output_size = (1,) * (v.ndim - 1) + (v.shape[v.ndim - 1])
        result = mnp.norm(v.view((-1, v.shape[v.ndim - 1])), pow1, 0).view(output_size)
    else:
        result = norm_except_dim(v.swapaxes(0, dim), pow1, dim).swapaxes(0, dim)
    return result


def _weight_norm(v, g, dim):
    """weight norm"""
    return v * (g / norm_except_dim(v, 2, dim))


class WNConv1d(nn.Conv1d):
    """conv1d with weight normalization"""

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        # add g and v as new parameters and express w as g/||v|| * v
        self.param_g = Parameter(Tensor(norm_except_dim(self.weight, 2, self.dim)))
        self.param_v = Parameter(Tensor(self.weight.data))
        delattr(self, 'weight')

    def construct(self, x):
        """construct conv1d with weight normalization"""
        x = self.expand_dims(x, 2)
        weight = _weight_norm(self.param_v, self.param_g, self.dim)
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)

        output = self.squeeze(output)
        return output
