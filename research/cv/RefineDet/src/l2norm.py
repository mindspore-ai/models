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
"""L2Normalization for RefineDet"""

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.initializer import Constant

class L2Norm(nn.Cell):
    """L2 Normalization for refinedet"""
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = ms.Parameter(Tensor(shape=self.n_channels, dtype=ms.float32, init=Constant(self.gamma)))
        self.norm = P.L2Normalize(axis=1, epsilon=self.eps)
        self.expand_dims = P.ExpandDims()

    def construct(self, x):
        """construct network"""
        x = self.norm(x)
        out = self.expand_dims(self.expand_dims(self.expand_dims(self.weight, 0), 2), 3).expand_as(x) * x
        return out
