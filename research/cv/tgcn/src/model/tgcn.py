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
T-GCN architecture

For detailed information, please refer to the paper below:
https://arxiv.org/pdf/1811.05320.pdf
"""
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer, XavierUniform, Constant
from .graph_conv import calculate_laplacian_with_self_loop


class TGCNGraphConvolution(nn.Cell):
    """
    T-GCN graph convolution layer
    """

    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.matmul = nn.MatMul()
        self.laplacian = Parameter(calculate_laplacian_with_self_loop(Tensor(adj, mstype.float32), self.matmul),
                                   name='laplacian', requires_grad=False)
        self.weights = Parameter(initializer(XavierUniform(), [self._num_gru_units + 1, self._output_dim],
                                             mstype.float32), name='weights')
        self.biases = Parameter(initializer(Constant(self._bias_init_value), [self._output_dim],
                                            mstype.float32), name='biases')

    def construct(self, inputs, hidden_state):
        """
        Calculate graph convolution outputs

        Args:
            inputs(Tensor): network inputs
            hidden_state(Tensor): hidden state

        Returns:
            outputs: TGCNGraphConvolution outputs
        """
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = P.Concat(axis=2)((inputs, hidden_state))
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(1, 2, 0)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape((num_nodes, (self._num_gru_units + 1) * batch_size))
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.matmul(self.laplacian, concatenation)
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape((num_nodes, self._num_gru_units + 1, batch_size))
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(2, 0, 1)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self._num_gru_units + 1))
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = self.matmul(a_times_concat, self.weights) + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs


class TGCNCell(nn.Cell):
    """
    T-GCN cell
    """

    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.adj = Parameter(Tensor(adj, mstype.float32), name='adj', requires_grad=False)
        self.graph_conv1 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.graph_conv2 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim)

    def construct(self, inputs, hidden_state):
        """
        Calculate hidden states

        Args:
            inputs(Tensor): network inputs
            hidden_state(Tensor): hidden state

        Returns:
            new_hidden_state: new hidden state
        """
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = P.Sigmoid()(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units), u (batch_size, num_nodes, num_gru_units)
        r, u = P.Split(axis=1, output_num=2)(concatenation)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = P.Tanh()(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state


class TGCN(nn.Cell):
    """
    T-GCN network
    """

    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.adj = Parameter(Tensor(adj, mstype.float32), name='adj', requires_grad=False)
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)

    def construct(self, inputs):
        """
        Calculate the final output

        Args:
            inputs(Tensor): network inputs

        Returns:
            output: TGCN output
        """
        batch_size, seq_len, num_nodes = inputs.shape
        hidden_state = P.Zeros()((batch_size, num_nodes * self._hidden_dim), mstype.float32)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output
