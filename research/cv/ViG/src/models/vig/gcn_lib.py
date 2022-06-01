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
"graph conv functions for vig"
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops


def pairwise_distance(x):
    """
    Compute pairwise distance.
    """
    x_inner = -2 * ops.matmul(x, ops.Transpose()(x, (0, 2, 1)))
    x_square = ops.ReduceSum(True)(ops.mul(x, x), axis=-1)
    return x_square + x_inner + ops.Transpose()(x_square, (0, 2, 1))


def dense_knn_matrix(x, k=16):
    """Get kNN based on the pairwise distance.
    """
    x = ops.Transpose()(x, (0, 2, 1, 3)).squeeze(-1)
    batch_size, n_points, _ = x.shape
    dist = pairwise_distance(x)
    _, nn_idx = ops.TopK()(-dist, k)
    center_idx = Tensor(np.arange(0, n_points), mstype.int32)
    center_idx = ops.Tile()(center_idx, (batch_size, k, 1))
    center_idx = ops.Transpose()(center_idx, (0, 2, 1))
    return ops.Stack(axis=0)((nn_idx, center_idx))


def batched_index_select(x, idx):
    """fetches neighbors features from a given neighbor idx.
    """
    batch_size, num_dims, num_vertices = x.shape[:3]
    k = idx.shape[-1]
    idx_base = Tensor(np.arange(0, batch_size), mstype.int32).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.view(-1)

    x = ops.Transpose()(x, (0, 2, 1, 3))
    feature = x.view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims)
    feature = ops.Transpose()(feature, (0, 3, 1, 2))
    return feature


class MRGraphConv2d(nn.Cell):
    """Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751).
    """

    def __init__(self, in_channels, out_channels, k=9, dilation=1, bias=True):
        super(MRGraphConv2d, self).__init__()
        self.k = k
        self.dilation = dilation
        self.nn = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, group=4, has_bias=bias)

    def construct(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1, 1)

        edge_index = dense_knn_matrix(x, self.k)
        edge_index = edge_index[:, :, :, ::self.dilation]

        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j = ops.ReduceMax(True)(x_j - x_i, -1)
        x = ops.Concat(axis=1)([x, x_j])
        return self.nn(x).view(b, -1, h, w)
