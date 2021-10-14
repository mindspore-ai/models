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
Graph convolution operation
"""
import mindspore.numpy as np
import mindspore.ops.operations as P
from mindspore import dtype as mstype


def calculate_laplacian_with_self_loop(matrix, matmul):
    """
    Calculate laplacian matrix with self loop

    Args:
        matrix(Tensor): input matrix
        matmul(MatMul): the MatMul operator for mixed precision

    Returns:
        normalized_laplacian: normalized laplacian matrix
    """
    matrix = matrix + P.Eye()(matrix.shape[0], matrix.shape[0], mstype.float32)
    row_sum = matrix.sum(1)
    d_inv_sqrt = P.Pow()(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_laplacian = matmul(matmul(matrix, d_mat_inv_sqrt).transpose(0, 1), d_mat_inv_sqrt)
    return normalized_laplacian
