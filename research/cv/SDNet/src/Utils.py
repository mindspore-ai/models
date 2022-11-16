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

import mindspore.nn as nn
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np


class L2_norm(nn.Cell):
    def __init__(self):
        super(L2_norm, self).__init__()
        self.eps = 1e-10
        self.sqrt = ops.Sqrt()
        self.sum = ops.ReduceSum()

    def construct(self, x):
        norm = self.sqrt(self.sum(x * x, 1) + self.eps)
        x = x / ops.ExpandDims()(norm, -1).expand_as(x)
        return x


def orthogonal(shape, gain=1):
    mul = ops.Mul()
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = Tensor.from_numpy(q).astype(mindspore.float32)
    weight = mul(q.reshape(shape), gain)
    return weight
