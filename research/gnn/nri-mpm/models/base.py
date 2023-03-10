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

import math
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import numpy as np
import mindspore.common.initializer as init

class MyBatchNorm1d(nn.Cell):
    """
    Batch Normalization layer over a 3D input.
    """
    def __init__(self, num_features: int, dim: int = -1):
        super(MyBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, use_batch_statistics=True)
        self.dim = dim

    def construct(self, x: Tensor) -> Tensor:
        y = x.view(-1, x.shape[self.dim])
        y = self.bn(y)
        return y.view(*x.shape)


class MLP(nn.Cell):
    """
    Two-layer fully-connected ELU net with batch norm.
    """
    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0.):
        """
        Parameters
        ----------
        n_in : int
            input dimension.
        n_hid : int
            dimension of hidden layers.
        n_out : int
            output dimension.
        do_prob : float, optional
            rate of dropout. The default is 0..
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Dense(n_in, n_hid)
        self.fc2 = nn.Dense(n_hid, n_out)
        self.bn1 = MyBatchNorm1d(n_hid)
        self.bn2 = MyBatchNorm1d(n_out)
        self.dropout = nn.Dropout(p=do_prob)
        self.elu = nn.ELU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.elu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.elu(self.fc2(x))
        x = self.bn2(x)
        return x


class LinAct(nn.Cell):
    """
    A linear layer with a non-linear activation function.
    """
    def __init__(self, n_in: int, n_out: int, do_prob: float = 0.0, act=None):
        """
        Parameters
        ----------
        n_in : int
            input dimension.
        n_out : int
            output dimension.
        do_prob : float, optional
            rate of dropout. The default is 0..
        act : TYPE, optional
            active function. The default is None.
        """
        super(LinAct, self).__init__()
        if act is None:
            act = nn.ReLU()
        self.model = nn.SequentialCell([
            nn.Dense(n_in, n_out),
            act,
            nn.Dropout(p=do_prob)])

    def construct(self, x: Tensor) -> Tensor:
        return self.model(x)


class SelfAtt(nn.Cell):
    """
    Self-attention.
    """
    def __init__(self, n_in: int, n_out: int):
        """
        Parameters
        ----------
        n_in : int
            input dimension.
        n_out : int
            output dimension.
        """
        super(SelfAtt, self).__init__()
        self.query, self.key, self.value = nn.CellList([
            nn.SequentialCell([nn.Dense(n_in, n_out), nn.Tanh()])
            for _ in range(3)])
        self.bmm = ops.BatchMatMul()
        self.softmax = ops.Softmax()
        self.scale = Tensor(n_out, ms.float32)

    def construct(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [..., size, dim].

        Returns
        -------
        out : Tensor
            [..., size, dim].
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # scaled dot product
        alpha = self.bmm(query, key.swapaxes(-1, -2)) / np.sqrt(self.scale)
        att = self.softmax(alpha)
        out = self.bmm(att, value)
        return out


class CNN(nn.Cell):
    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0.):
        """
        Parameters
        ----------
        n_in : int
            input dimension.
        n_hid : int
            dimension of hidden layers.
        n_out : int
            output dimension.
        do_prob : float, optional
            rate of dropout. The default is 0..
        """
        super(CNN, self).__init__()
        self.cnn = nn.SequentialCell([
            nn.Conv1d(n_in, n_hid, kernel_size=5, pad_mode="valid", has_bias=True,
                      weight_init=init.Normal(math.sqrt(2./(5*n_hid))),
                      bias_init=init.Constant(0.1)),
            nn.ReLU(),
            MyBatchNorm1d(n_hid, dim=1),
            nn.Dropout(p=do_prob),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(n_hid, n_hid, kernel_size=5, pad_mode="valid", has_bias=True,
                      weight_init=init.Normal(math.sqrt(2./(5*n_hid))),
                      bias_init=init.Constant(0.1)),
            nn.ReLU(),
            MyBatchNorm1d(n_hid, dim=1)
            ])
        self.out = nn.Conv1d(n_hid, n_out, kernel_size=1, pad_mode="valid", has_bias=True,
                             weight_init=init.Normal(math.sqrt(2./n_out)),
                             bias_init=init.Constant(0.1))
        self.att = nn.Conv1d(n_hid, 1, kernel_size=1, pad_mode="valid", has_bias=True,
                             weight_init=init.Normal(math.sqrt(2.0)),
                             bias_init=init.Constant(0.1))
        self.softmax = nn.Softmax(2)

    def construct(self, inputs: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inputs : Tensor
            [batch * E, dim, step], raw edge representation at each step.

        Returns
        -------
        edge_prob: Tensor
            [batch * E, dim], edge representations over all steps with the step-dimension reduced.
        """
        x = self.cnn(inputs)
        pred = self.out(x)
        attention = self.softmax(self.att(x))
        edge_prob = (pred * attention).mean(axis=2)
        return edge_prob


class GNN(nn.Cell):
    """
    Reimplementation of the Message-Passing class to allow more flexibility.
    """

    def aggregate(self, msg: Tensor, idx: Tensor, size: int, agg: str = "mean") -> Tensor:
        """
        Parameters
        ----------
        msg : Tensor
            [E, ..., dim * 2].
        idx : Tensor
            [E].
        size : int
            number of nodes.
        agg : str, optional
            only 3 types of aggregation are supported: 'add', 'mean' or 'max'. The default is "mean".

        Returns
        -------
        Tensor
            aggregated node embeddings.
        """
        msg = msg.view(size, -1, *msg.shape[1:])
        out = msg.mean(1)
        return out

    def message(self, x: Tensor, es: Tensor):
        """
        Parameters
        ----------
        x : Tensor
            [node, ..., dim], node embeddings.
        es : Tensor
            [2, E], edge list.

        Returns
        -------
        Tensor
            msg: [E, ..., dim * 2], edge embeddings.
        Tensor
            col: [E], indices of msg.
        int
            size: number of nodes.
        """
        col, row = es
        x_i, x_o = x[row], x[col]
        msg = ops.Concat(-1)((x_i, x_o))
        return msg, col, len(x)


class GRUCell(nn.Cell):
    def __init__(self, dim_in: int, dim_hid: int, bias: bool = True):
        """
        Parameters
        ----------
        dim_in : int
            input dimension.
        dim_hid : int
            dimension of hidden layers.
        bias : bool, optional
            adding a bias term or not. The default is True.
        """
        super(GRUCell, self).__init__()
        self.hidden = nn.CellList([
            nn.Dense(dim_hid, dim_hid, has_bias=bias)
            for _ in range(3)])
        self.input = nn.CellList([
            nn.Dense(dim_in, dim_hid, has_bias=bias)
            for _ in range(3)])
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def construct(self, inputs: Tensor, hidden: Tensor, state: Tensor = None) -> Tensor:
        """
        Parameters
        ----------
        inputs : Tensor
            [..., dim].
        hidden : Tensor
            [..., dim].
        state : Tensor, optional
            [..., dim]. The default is None.
        """
        r = self.sigmoid(self.input[0](inputs) + self.hidden[0](hidden))
        i = self.sigmoid(self.input[1](inputs) + self.hidden[1](hidden))
        n = self.tanh(self.input[2](inputs) + r * self.hidden[2](hidden))
        if state is None:
            state = hidden
        output = (1 - i) * n + i * state
        return output
