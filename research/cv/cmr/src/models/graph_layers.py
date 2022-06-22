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

import math

import mindspore
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import ops
from mindspore import Parameter, Tensor
from mindspore.common import initializer


class GraphConvolution(nn.Cell):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, adjmat, bias=True):
        """
        :param adjmat: adjmat is a dense matrix
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adjmat = adjmat
        stdv = 6. / math.sqrt(in_features + out_features)

        self.weight = Parameter(Tensor(shape=(in_features, out_features),
                                       init=initializer.Uniform(stdv), dtype=mindspore.float32))
        if bias:
            self.bias = Parameter(Tensor(shape=out_features,
                                         init=initializer.Uniform(stdv), dtype=mindspore.float32))

        self.matmul = P.MatMul()
        self.stack = P.Stack(axis=0)

    def construct(self, x):
        if x.ndim == 2:
            support = self.matmul(x, self.weight)
            output = self.matmul(self.adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        output = []
        for i in range(x.shape[0]):
            support = self.matmul(x[i], self.weight)
            output.append(self.matmul(self.adjmat, support))

        output = self.stack(output)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GraphLinear(nn.Cell):
    """
        Generalization of 1x1 convolutions on Graphs
    """
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W = Parameter(Tensor(shape=(self.out_channels, self.in_channels),
                                  init=initializer.Uniform(w_stdv), dtype=mindspore.float32))
        self.b = Parameter(Tensor(shape=self.out_channels, init=initializer.Uniform(w_stdv),
                                  dtype=mindspore.float32))

        self.matmul = P.MatMul()

    def construct(self, x):
        # [1, out_channels(1024), in_channels(2051)] @ [batch_size, 2051, 1723]
        if x.ndim == 4:
            x = x[:, :, :, 0]
        return ops.matmul(self.W[None, :], x) + self.b[None, :, None]


class GraphResBlock(nn.Cell):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, A):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, A)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        self.pre_norm = nn.GroupNorm(in_channels // 8, in_channels)
        self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
        self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
        self.relu = nn.ReLU()

    def construct(self, x):
        # x shape: (16, 512, 1723)
        if x.ndim == 3:
            x = x[:, :, :, None]
        y = self.pre_norm(x)[:, :, :, 0]
        x = x[:, :, :, 0]
        y = self.relu(y)
        y = self.lin1(y)

        if y.ndim == 3:
            y = y[:, :, :, None]
        y = self.norm1(y)[:, :, :, 0]
        y = self.relu(y)
        y = self.conv(y.transpose((0, 2, 1))).transpose((0, 2, 1))

        if y.ndim == 3:
            y = y[:, :, :, None]
        y = self.norm2(y)[:, :, :, 0]
        y = self.relu(y)
        y = self.lin2(y)
        if self.in_channels != self.out_channels:
            x = self.skip_conv(x)
        return (x + y)[:, :, :, None]
