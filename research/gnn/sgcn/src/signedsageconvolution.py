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
"""Layer classes."""
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
import mindspore.numpy as mnp
from mindspore.common.initializer import Uniform
from mindspore.ops.primitive import constexpr


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


@constexpr  # 当前不支持在网络里构造Tensor 可以通过@constexpr装饰器修饰函数，在函数里生成Tensor
def range_tensor(start, end):
    return Tensor(np.arange(start, end), ms.int32)


@constexpr  # input_x parameter -> tensor
def construct_tensor(input_x):
    return Tensor(input_x.asnumpy())


def add_self_loops(edge_index, num_nodes):
    N = maybe_num_nodes(edge_index, num_nodes)
    loop_index = mnp.arange(0, N)
    loop_index = ops.ExpandDims()(loop_index, 0)
    loop_index = ops.Tile()(loop_index, (2, 1)).astype("int32")
    edge_index = ops.Concat(1)((edge_index, loop_index))
    return edge_index


@constexpr  # 当前不支持在网络里构造Tensor 可以通过@constexpr装饰器修饰函数，在函数里生成Tensor
def loops_out(start, end):
    return Tensor(mnp.zeros((start, end)))
    # return Parameter(Tensor(mnp.zeros((start, end))), requires_grad = False)


# pytorch scatter_add平替版 针对bitcoin数据集有效。
def ms_scatter_add(src, index, dim_size, dim=0):
    out = loops_out(dim_size, src.shape[1])
    # out = Tensor(mnp.zeros((dim_size, src.shape[1])))
    out = ops.ScatterAdd()(out, index.astype("int32"), src)
    return out


# pytorch scatter_mean平替版。
def ms_scatter_mean2(src, index, dim_size, dim=0):
    shape = (dim_size, src.shape[1])
    indices = ops.ExpandDims()(index.astype("int32"), 0).T
    res = ops.ScatterNd()(indices, src, shape)
    tag = mnp.ones((src.shape[0], src.shape[1]))
    tag = ops.ScatterNd()(indices, tag, shape)
    tag = tag + (tag == 0).astype("int32")
    out = ops.Div()(res, tag)
    return out


class SignedSAGEConvolution(nn.Cell):
    """
    Abstract Signed SAGE convolution class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param norm_embed: Normalize embedding -- boolean.
    :param bias: Add bias or no.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 name='base_',
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(SignedSAGEConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.matmul = nn.MatMul().to_float(ms.float16)
        self.l2_normalize = ops.L2Normalize(epsilon=1e-12, axis=-1)
        self.concat = ops.Concat(axis=1)
        self.weight_tensor = Tensor(shape=[self.in_channels, out_channels], dtype=ms.float32,
                                    init=Uniform(scale=1.0 / math.sqrt(self.in_channels)))
        self.weight = Parameter(self.weight_tensor, name=name + 'weight')
        print('self.weight', type(self.weight), self.weight.shape, self.weight.dtype)
        self.weight_data = list(self.weight)
        if bias:
            tmp = Tensor(shape=[out_channels], dtype=ms.float32, init=Uniform(scale=1.0 / math.sqrt(self.in_channels)))
            self.bias = Parameter(tmp, name=name + 'bias')
        else:
            self.register_parameter("bias", None)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class SignedSAGEConvolutionBase(SignedSAGEConvolution):
    """
    Base Signed SAGE class for the first layer of the model.
    """

    def construct(self, x, edge_index):
        """
        Forward propagation pass with features an indices.
        Args:
            x(Tensor): Indices of edges.
            edge_index(Tensor): Indices of edges.

        Returns:
            out(Tensor): Abstract convolved features.
        """
        row, col = edge_index
        if self.norm:
            out = ms_scatter_mean2(x[col], row, dim_size=x.shape[0])
        else:
            out = ms_scatter_add(x[col], row, dim_size=x.shape[0])
        out = self.concat((out, x))
        out = self.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = self.l2_normalize(out)
        return out


class SignedSAGEConvolutionDeep(SignedSAGEConvolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """

    def construct(self, x_1, x_2, edge_index_pos, edge_index_neg):
        """
        Forward propagation pass with features an indices.
        Args:
            x_1(Tensor): Features for left hand side vertices.
            x_2(Tensor): Features for right hand side vertices.
            edge_index_pos(Tensor): Positive indices.
            edge_index_neg(Tensor): Negative indices.

        Returns:
            out(Tensor): Abstract convolved features.
        """
        edge_index_pos = add_self_loops(edge_index_pos, num_nodes=x_1.shape[0])
        edge_index_neg = add_self_loops(edge_index_neg, num_nodes=x_2.shape[0])
        row_pos, col_pos = edge_index_pos
        row_neg, col_neg = edge_index_neg
        if self.norm:
            out_1 = ms_scatter_mean2(x_1[col_pos], row_pos, dim=0, dim_size=x_1.shape[0])
            out_2 = ms_scatter_mean2(x_2[col_neg], row_neg, dim=0, dim_size=x_2.shape[0])
        else:
            out_1 = ms_scatter_add(x_1[col_pos], row_pos, dim=0, dim_size=x_1.shape[0])
            out_2 = ms_scatter_add(x_2[col_neg], row_neg, dim=0, dim_size=x_2.shape[0])

        out = self.concat((out_1, out_2, x_1))
        out = self.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = self.l2_normalize(out)
        return out
