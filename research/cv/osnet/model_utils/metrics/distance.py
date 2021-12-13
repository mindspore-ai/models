# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Compute feature distance."""

import numpy as np


import mindspore
from mindspore import Tensor
import mindspore.ops as ops

def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.
    """
    # check input
    assert isinstance(input1, Tensor)
    assert isinstance(input2, Tensor)
    assert input1.ndim == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.ndim
    )
    assert input2.ndim == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.ndim
    )
    assert input1.shape[1] == input2.shape[1]

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.shape[0], input2.shape[0]

    shape_tensor1 = Tensor(np.zeros((m, n), dtype=np.float32))
    shape_tensor2 = Tensor(np.zeros((n, m), dtype=np.float32))
    op_pow = ops.Pow()

    mat1 = op_pow(input1, 2).sum(axis=1, keepdims=True).expand_as(shape_tensor1)
    mat2 = op_pow(input2, 2).sum(axis=1, keepdims=True).expand_as(shape_tensor2).T
    distmat = mat1 + mat2
    matmul = ops.MatMul(False, True)
    cast = ops.Cast()
    input1 = cast(input1, mindspore.float16)
    input2 = cast(input2, mindspore.float16)
    output = cast(matmul(input1, input2), mindspore.float32)
    distmat = distmat - 2 * output

    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    l2_normalize = ops.L2Normalize(axis=1)
    input1_normed = l2_normalize(input1)
    input2_normed = l2_normalize(input2)
    matmul = ops.MatMul(False, True)
    distmat = 1 - matmul(input1_normed, input2_normed)
    return distmat
