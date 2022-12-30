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

import mindspore as ms
from mindspore import nn, Tensor, ops
from mindspore import numpy as np

def transpose(x: Tensor, size: int) -> Tensor:
    """
    Transpose the edge features x.
    """
    index = transpose_id(size)
    return x[index]


def transpose_id(size: int) -> Tensor:
    """
    Return the edge list corresponding to a transposed adjacency matrix.
    """
    idx = np.arange(size * (size - 1))
    ii = np.floor_divide(idx, size-1)
    jj = np.mod(idx, size-1)
    jj = np.multiply(jj, np.less(jj, ii)) + np.multiply(jj+1, np.greater_equal(jj, ii))
    index = np.multiply(jj, size-1) + np.multiply(ii, np.less(ii, jj)) + np.multiply(ii-1, np.greater(ii, jj))
    return index

class GumbelSoftmax(nn.Cell):
    def __init__(self):
        super().__init__()
        self.zero = Tensor(0, ms.float32)
        self.one = Tensor(1, ms.float32)

    def construct(self, logits: Tensor, tau: float = 1., hard: bool = False, eps: float = 1e-10, dim: int = -1):
        U = ops.uniform(logits.shape, self.zero, self.one)
        gumbel_noise = -ops.log(-ops.log(U + eps) + eps)
        y = logits + gumbel_noise
        y_soft = ops.Softmax()(y / tau)
        if hard:
            index = y_soft.argmax(dim)
            y_hard = ops.OneHot()(index, 2, self.one, self.zero)
            ret = y_hard
        else:
            ret = y_soft
        return ret
