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
from mindspore import nn, ops, Tensor
import mindspore.common.initializer as init
from models.base import MLP, SelfAtt, CNN, GNN, GRUCell
from utils.helper import GumbelSoftmax

class AttENC(GNN):
    """
    Encoder using the relation interaction mechanism implemented by self-attention.
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
            output dimension, i.e., number of edge types.
        do_prob : float, optional
            rate of dropout. The default is 0..
        factor : bool, optional
            using a factor graph or not. The default is True.
        reducer : str, optional
            using an MLP or a CNN to reduce edge representations over multiple steps. The default is "mlp".
        option : str, optional
            "intra": using the intra-edge interaction operation,
            "inter": using the inter-edge interaction operation,
            "both": using both operations.
            The default is "both".
        """
        super(AttENC, self).__init__()
        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
        self.n2e_i = MLP(n_hid, n_hid, n_hid, do_prob)
        self.e2n = MLP(n_hid, n_hid, n_hid, do_prob)
        self.n2e_o = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        # self-attention for both intra-edge and inter-edge operations
        self.intra_att = SelfAtt(n_hid, n_hid)
        self.inter_att = SelfAtt(n_hid, n_hid)
        self.fc_out = nn.Dense(n_hid * 2, n_out,
                               weight_init=init.XavierUniform(), bias_init=init.Constant(0.1))

    def reduce_cnn(self, x: Tensor, es: Tensor):
        """
        Parameters
        ----------
        x : Tensor
            [node, batch, step, dim].
        es : Tensor
            [2, E].

        Returns
        -------
        z : Tensor
            [E, batch, dim]
        col : Tensor
            [E]
        size : int
        """
        # z: [E, batch, step, dim * 2]
        z, col, size = self.message(x, es)
        z = z.swapaxes(3, 2)
        z = z.view(-1, z.shape[2], z.shape[3])
        z = self.cnn(z)
        z = z.view(len(col), x.shape[1], -1)
        return z, col, size

    def intra_es(self, z: Tensor, size: int) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor
            [E, batch, dim].
        size : int

        Returns
        -------
        zs : Tensor
            [size, size - 1, batch, dim].
        """
        E, batch, dim = z.shape
        # zz: [size, size - 1, batch ,dim]
        zz = z.view(E//(size-1), size-1, batch, dim)
        # zz: [size, batch, size - 1, dim]
        zz = zz.swapaxes(1, 2)
        zs = self.intra_att(zz)
        zs = zs.swapaxes(1, 2)
        return zs

    def inter_es(self, zs: Tensor, size: int) -> Tensor:
        """
        Parameters
        ----------
        zs : Tensor
            [size, size - 1, batch, dim].
        size : int

        Returns
        -------
        hs : Tensor
            [size, size - 1, batch, dim].
        """
        # mean pooling to get the overall embedding of all incoming edges
        h = zs.mean(1)
        h = h.swapaxes(0, 1)
        hs = self.inter_att(h)
        hs = hs.swapaxes(0, 1)
        hs = ops.expand_dims(hs, 1)
        hs = ops.tile(hs, (1, size-1, 1, 1))
        return hs

    def construct(self, x: Tensor, es: Tensor) -> Tensor:
        """
        Given the historical node states, output the K-dimension edge representations ready for relation prediction.

        Parameters
        ----------
        x : Tensor
            [batch, step, node, dim], node representations.
        es : Tensor
            [2, E], edge list.

        Returns
        -------
        z : Tensor
            [E, batch, dim], edge representations.
        """
        x = x.transpose((2, 0, 1, 3))
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        z, col, size = self.reduce_cnn(x, es)
        z = self.n2e_i(z)
        z_skip = z
        h = self.aggregate(z, col, size)
        h = self.e2n(h)
        z, _, _ = self.message(h, es)
        # skip connection
        z = ops.Concat(2)((z, z_skip))
        z = self.n2e_o(z)
        _, batch, dim = z.shape
        zs = self.intra_es(z, size)
        hs = self.inter_es(zs, size)
        zs = ops.Concat(-1)((zs, hs))
        z = zs.view(-1, batch, dim * 2)
        z = self.fc_out(z)
        return z


class RNNDEC(GNN):
    """
    RNN decoder with spatio-temporal message passing mechanisms.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float = 0., skip_first: bool = False):
        """
        Parameters
        ----------
        n_in_node : int
            input dimension.
        edge_types : int
            number of edge types.
        msg_hid, msg_out, n_hid: int
            dimension of different hidden layers.
        do_prob : float, optional
            rate of dropout. The default is 0..
        skip_first : bool, optional
            setting the first type of edge as non-edge or not, if yes,
            the first type of edge will have no effect. The default is False.
        option : str, optional
            "both": using both node-level and edge-level spatio-temporal message passing operations,
            "node": using node-level the spatio-temporal message passing operation,
            "edge": using edge-level the spatio-temporal message passing operation.
            The default is "both".
        """
        super(RNNDEC, self).__init__()
        self.msgs = nn.CellList([
            nn.SequentialCell([
                nn.Dense(2 * n_in_node, msg_hid),
                nn.ReLU(),
                nn.Dropout(p=do_prob),
                nn.Dense(msg_hid, msg_out),
                nn.ReLU()])
            for _ in range(edge_types)])
        self.out = nn.SequentialCell([
            nn.Dense(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(p=do_prob),
            nn.Dense(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(p=do_prob),
            nn.Dense(n_hid, n_in_node)])
        self.gru_edge = GRUCell(n_hid, n_hid)
        self.gru_node = GRUCell(n_hid + n_in_node, n_hid + n_in_node)
        self.msg_out = msg_out
        self.skip_first = skip_first

    def move(self, x: Tensor, es: Tensor, z: Tensor, h_node: Tensor = None, h_edge: Tensor = None):
        """
        Parameters
        ----------
        x : Tensor
            [node, batch, step, dim].
        es : Tensor
            [2, E].
        z : Tensor
            [E, batch, K].
        h_node : Tensor, optional
            [node, batch, step, dim], hidden states of nodes. The default is None.
        h_edge : Tensor, optional
            [E, batch, step, dim], hidden states of edges. The default is None.

        Returns
        -------
        out : Tensor
            [node, batch, step, dim], future node states
        msgs : Tensor
            [E, batch, step, dim], hidden states of edges
        cat : Tensor
            [node, batch, step, dim], hidden states of nodes
        """
        # z: [E, batch, K] -> [E, batch, step, K]
        z = ops.expand_dims(z, 2)
        z = ops.tile(z, (1, 1, x.shape[2], 1))
        msg, col, size = self.message(x, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)
        if self.skip_first:
            norm -= 1

        msgs_l = []
        for i in range(idx, len(self.msgs)):
            msgs_l.append(self.msgs[i](msg) * ops.expand_dims(z[:, :, :, i], -1) / norm)
        msgs = ops.reduce_sum(ops.stack(msgs_l), 0)

        if h_edge is not None:
            msgs = self.gru_edge(msgs, h_edge)
        # aggregate all msgs from the incoming edges
        msg = self.aggregate(msgs, col, size)
        # skip connection
        cat = ops.Concat(-1)((x, msg))
        if h_node is not None:
            cat = self.gru_node(cat, h_node)
        delta = self.out(cat)
        return x + delta, cat, msgs

    def construct(self, x: Tensor, z: Tensor, es: Tensor, M: int = 1) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [batch, step, node, dim], historical node states.
        z : Tensor
            [E, batch, K], distribution of edge types.
        es : Tensor
            [2, E], edge list.
        M : int, optional
            number of steps to predict. The default is 1.

        Returns
        -------
        out : Tensor
            future node states.
        """
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.transpose((2, 0, 1, 3))
        # only take m-th timesteps as starting points (m: pred_steps).
        x_m = x[:, :, 0::M, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict m steps.
        xs = []
        h_node, h_edge = None, None
        for _ in range(M):
            x_m, h_node, h_edge = self.move(x_m, es, z, h_node, h_edge)
            xs.append(x_m)

        node, batch, _, dim = xs[0].shape
        x_hat = ops.Stack(-2)(xs)
        x_hat = x_hat.view(node, batch, -1, dim)
        x_hat = x_hat.transpose((1, 2, 0, 3))
        return x_hat[:, :(x.shape[2] - 1)]


class NRIModel(nn.Cell):
    """
    Auto-encoder.
    """
    def __init__(self, dim, n_hid, edge_type, do_prob, skip_first, size, es):
        """
        Parameters
        ----------
        encoder : nn.Cell
            an encoder inferring relations.
        decoder : nn.Cell
            an decoder predicting future states.
        es : Tensor
            edge list.
        size : int
            number of nodes.
        """
        super(NRIModel, self).__init__()
        self.enc = AttENC(dim, n_hid, edge_type, do_prob)
        self.dec = RNNDEC(dim, edge_type, n_hid, n_hid, n_hid, do_prob, skip_first)
        self.gumbel_softmax = GumbelSoftmax()
        self.es = Tensor(es, dtype=ms.int32)
        self.size = size

    def construct(self, states_enc: Tensor, states_dec: Tensor,
                  hard: bool = False, M: int = 10):
        """
        Parameters
        ----------
        states_enc : Tensor
            [batch, step_enc, node, dim], input node states for the encoder.
        states_dec : Tensor
            [batch, step_dec, node, dim], input node states for the decoder.
        hard : bool, optional
            predict one-hot representation of relations or its continuous relaxation. The default is False.
        p : bool, optional
            return the distribution of relations or not. The default is False.
        M : int, optional
            number of steps to predict. The default is 1.
        tosym : bool, optional
            impose hard constraint to inferred relations or not. The default is False.

        Returns
        -------
        output : Tensor
            [batch, step, node, dim].
        prob : Optional[Tensor]
            [E, batch, K]
        """
        logits = self.enc(states_enc, self.es)
        edges = self.gumbel_softmax(logits, tau=0.5, hard=hard)
        output = self.dec(states_dec, edges, self.es, M)
        prob = ops.Softmax()(logits)
        prob = prob.swapaxes(0, 1)
        return output, prob
