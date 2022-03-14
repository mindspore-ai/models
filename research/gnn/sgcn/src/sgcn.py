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
"""SGCN runner."""
import time

import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as mnp
from mindspore import ops
from mindspore import save_checkpoint
from mindspore.common.initializer import XavierUniform
from mindspore.common.initializer import Zero
from mindspore.ops.primitive import constexpr
from sklearn.model_selection import train_test_split

from src.metrics import TrainNetWrapper
from src.ms_utils import calculate_auc
from src.ms_utils import maybe_num_nodes
from src.ms_utils import setup_features
from src.signedsageconvolution import SignedSAGEConvolutionBase
from src.signedsageconvolution import SignedSAGEConvolutionDeep


@constexpr
def ms_isin(a, b):
    """
    Calculates elements from a which contains in b.
    Args:
        a: Input array.
        b: The values against which to test each value of input array.

    Returns:
        Tensor(np.isin(a.asnumpy(), b.asnumpy()), mstype.bool_)
    """
    return Tensor(np.isin(a.asnumpy(), b.asnumpy()), mstype.bool_)


@constexpr
def construct_tensor(size_rest, num_nodes):
    """
    Create tensor with integers from the uniform distribution.
    Args:
        size_rest: Output shape.
        num_nodes: Upper boundary of the output interval.

    Returns:
         Random integers from the uniform distribution.
    """
    minval = Tensor(0, mstype.int32)
    maxval = Tensor(num_nodes, mstype.int32)
    tmp = Tensor(ops.UniformInt(seed=10)((size_rest,), minval, maxval))
    return tmp


@constexpr
def range_tensor(start, end):
    """
    Create range tensor.
    Args:
        start: Min value.
        end: Max values.

    Returns:
        Tensor(np.arange(start, end), mstype.int32)
    """
    return Tensor(np.arange(start, end), mstype.int32)


@constexpr
def ms_nonzero(a):
    """
    Create tensor with the indices of the elements that are non-zero.
    Args:
        a: Input array.

    Returns:
        Tensor(res, dtype=mstype.int32).squeeze(axis=0)
    """
    res = a.asnumpy().nonzero()
    if res[0].shape[0] == 0:
        return res[0]
    return Tensor(res, dtype=mstype.int32).squeeze(axis=0)


@constexpr
def ms_appendindex(rest_index, rest, res):
    """ms_appendindex"""
    for temp in rest_index:
        res.append(int((rest[int(temp)]).asnumpy()))
    return Tensor(res)


class SignedGraphConvolutionalNetwork(nn.Cell):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    SGCN Initialization.
    """
    def __init__(self, x, norm, norm_embed, bias):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        self.X = x
        self.h_pos, self.h_neg = [], []
        self.tanh = ops.Tanh()
        self.op = ops.Concat(axis=1)
        self.norm = norm
        self.norm_embed = norm_embed
        self.bias = bias
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers.
        Assigning Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = [96, 64, 32]
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SignedSAGEConvolutionBase(
            self.X.shape[1]*2,
            self.neurons[0],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.negative_base_aggregator = SignedSAGEConvolutionBase(
            self.X.shape[1]*2,
            self.neurons[0],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.positive_aggregator_1 = SignedSAGEConvolutionDeep(
            3 * self.neurons[0],
            self.neurons[1],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.positive_aggregator_2 = SignedSAGEConvolutionDeep(
            3 * self.neurons[1],
            self.neurons[2],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.negative_aggregator_1 = SignedSAGEConvolutionDeep(
            3 * self.neurons[0],
            self.neurons[1],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.negative_aggregator_2 = SignedSAGEConvolutionDeep(
            3 * self.neurons[1],
            self.neurons[2],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.regression_weights = Parameter(
            Tensor(shape=(4 * self.neurons[-1], 3), dtype=mstype.float32, init=XavierUniform(gain=1.0))
        )
        self.regression_bias = Parameter(Tensor(shape=3, dtype=mstype.float32, init=Zero()))

    def construct(self, removed_pos, removed_neg):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        Args:
            removed_pos(Tensor): Positive edges without self loops.
            removed_neg(Tensor): Negative edges without self loops.

        Returns:
            z(Tensor): Hidden vertex representations.
        """
        h_pos, h_neg = [], []
        h_pos.append(self.tanh(self.positive_base_aggregator(self.X, removed_pos)))
        h_neg.append(self.tanh(self.negative_base_aggregator(self.X, removed_neg)))
        h_pos.append(self.tanh(self.positive_aggregator_1(h_pos[0], h_neg[0], removed_pos, removed_neg)))
        h_neg.append(self.tanh(self.negative_aggregator_1(h_neg[0], h_pos[0], removed_pos, removed_neg)))
        h_pos.append(self.tanh(self.positive_aggregator_2(h_pos[1], h_neg[1], removed_pos, removed_neg)))
        h_neg.append(self.tanh(self.negative_aggregator_2(h_neg[1], h_pos[1], removed_pos, removed_neg)))
        z = self.op((h_pos[-1], h_neg[-1]))
        return z


class SignedGCNTrainer:
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        """
        self.args = args
        self.edges = edges
        self.setup_logs()
        self.reshape = ops.Reshape()
        self.size = ops.Size()

    def setup_dataset(self):
        """
        Returns:
            self.X(Tensor): Dataset.
            self.positive_edges(Tensor): Positive edges for training.
            self.negative_edges(Tensor): Negative edges for training.
            self.test_positive_edges(Tensor): Positive edges for testing.
            self.test_negative_edges(Tensor): Negative edges for testing.
        """
        self.positive_edges, self.test_positive_edges = train_test_split(self.edges["positive_edges"],
                                                                         test_size=self.args.test_size, random_state=1)

        self.negative_edges, self.test_negative_edges = train_test_split(self.edges["negative_edges"],
                                                                         test_size=self.args.test_size, random_state=1)
        self.ecount = len(self.positive_edges + self.negative_edges)

        self.X = setup_features(self.args,
                                (self.positive_edges), #list
                                (self.negative_edges), #list
                                self.edges["ncount"]) #int
        self.X = mnp.array((self.X).tolist())
        self.positive_edges = mnp.array(self.positive_edges, dtype=mstype.int32).T
        self.negative_edges = mnp.array(self.negative_edges, dtype=mstype.int32).T
        self.y = mnp.array([0 if i < int(self.ecount / 2) else 1 for i in range(self.ecount)] + [2] * (self.ecount * 2))
        self.y = mnp.array(self.y, mnp.int32)
        self.X = mnp.array(self.X, mnp.float32)
        print('self.positive_edges', self.positive_edges.shape, type(self.positive_edges))
        print('self.negative_edges', self.negative_edges.shape, type(self.negative_edges))
        print('self.y', self.y.shape, type(self.y))
        print(self.positive_edges.dtype, self.negative_edges.dtype, self.y.dtype)
        return self.X, self.positive_edges, self.negative_edges, self.test_positive_edges, self.test_negative_edges

    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        self.model = SignedGraphConvolutionalNetwork(self.X, self.args.norm, self.args.norm_embed, self.args.bias)
        self.removed_pos = self.remove_self_loops(self.positive_edges)
        self.removed_neg = self.remove_self_loops(self.negative_edges)
        train_z = self.model(self.removed_pos, self.removed_neg)
        num_nodes = train_z.shape[0]
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.args.learning_rate,
                                 weight_decay=self.args.weight_decay)
        self.epochs = self.args.epochs
        train_net = TrainNetWrapper(self.model, self.y, weight_decay=self.args.weight_decay,
                                    learning_rate=self.args.learning_rate, lamb=self.args.lamb)
        best_auc, best_f1 = 0, 0
        t0 = time.time()
        for epoch in range(self.epochs):
            t = time.time()
            train_net.set_train()
            regression_positive_i, regression_positive_j, regression_positive_k, = \
                self.structured_sampling(self.positive_edges, num_nodes)
            regression_negative_i, regression_negative_j, regression_negative_k, = \
                self.structured_sampling(self.negative_edges, num_nodes)
            positive_i, positive_j, positive_k, = self.structured_sampling(self.positive_edges, num_nodes)
            negative_i, negative_j, negative_k, = self.structured_sampling(self.negative_edges, num_nodes)
            train_loss = train_net(self.removed_pos, self.removed_neg,
                                   regression_positive_i, regression_positive_j, regression_positive_k,
                                   regression_negative_i, regression_negative_j, regression_negative_k,
                                   positive_i, positive_j, positive_k, negative_i, negative_j, negative_k)
            auc, f1 = self.score_model(epoch)
            if self.args.rank_log_save_ckpt_flag:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", train_loss,
                      "time=", (time.time() - t), "auc=", auc, "f1=", f1)
                if auc > best_auc:
                    best_auc = auc
                    save_checkpoint(self.model, self.args.checkpoint_file + '_auc.ckpt')
                    print('Best AUC checkpoint has been saved.')
                if f1 > best_f1:
                    best_f1 = f1
                    save_checkpoint(self.model, self.args.checkpoint_file + '_f1.ckpt')
                    print('Best F1-Score checkpoint has been saved.')
        if self.args.rank_log_save_ckpt_flag:
            print('Training finished! The best AUC and F1-Score is:', best_auc, best_f1, 'Total time:',
                  time.time() - t0)

    def structured_sampling(self, edge_index, num_nodes=None):
        """
        Samples a negative edge for every positive edge
        Args:
            edge_index (LongTensor): The edge indices.
            num_nodes(Int): Number of nodes.

        Returns:
            i(Int): rows of edge indices.
            j(Int): columns of edge indices.
            k(Int): A tensor with given values.
        """
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        i, j = edge_index
        idx_1 = i * num_nodes + j
        k = construct_tensor(i.shape[0], num_nodes)
        idx_2 = i * num_nodes + k
        mask = ms_isin(idx_2, idx_1).squeeze()
        rest = ms_nonzero(mask)
        rest = self.reshape(rest, (-1,))
        while self.size(rest) > 0:
            tmp = construct_tensor(self.size(rest), num_nodes)
            idx_3 = i[rest] * num_nodes + tmp
            mask = (ms_isin(idx_3, idx_1).squeeze())
            k_np = k.asnumpy()
            k_np[rest.asnumpy()] = tmp.asnumpy()
            k = Tensor(k_np)
            res = []
            rest_index = ms_nonzero(mask)
            if rest_index.shape[0] == 0:
                break
            rest_index = ops.Reshape()(rest_index, (-1,)).asnumpy()
            rest = ms_appendindex(rest_index, rest, res)
        return i, j, k

    def remove_self_loops(self, edge_index):
        """
        remove self loops
        Args:
            edge_index (LongTensor): The edge indices.

        Returns:
            Tensor(edge_index): removed self loops
        """
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index.asnumpy()[:, mask.asnumpy()]
        return Tensor(edge_index)

    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        Args:
            epoch (LongTensor): Training epochs.

        Returns:
            auc(Float32): AUC result.
            f1(Float32): F1-Score result.
        """
        self.train_z = self.model(self.removed_pos, self.removed_neg)
        score_positive_edges = mnp.array(self.test_positive_edges, dtype=mnp.int32).T
        score_negative_edges = mnp.array(self.test_negative_edges, dtype=mnp.int32).T
        test_positive_z = ops.Concat(axis=1)((self.train_z[score_positive_edges[0, :], :],
                                              self.train_z[score_positive_edges[1, :], :]))
        test_negative_z = ops.Concat(axis=1)((self.train_z[score_negative_edges[0, :], :],
                                              self.train_z[score_negative_edges[1, :], :]))
        scores = ops.matmul(ops.Concat(axis=0)((test_positive_z, test_negative_z)),
                            self.model.regression_weights) + self.model.regression_bias
        probability_scores = ops.Exp()(ops.Softmax(axis=1)(scores))
        predictions = probability_scores[:, 0] / probability_scores[:, 0: 2].sum(1)
        predictions = predictions.asnumpy()
        targets = [0] * len(self.test_positive_edges) + [1] * len(self.test_negative_edges)
        auc, f1 = calculate_auc(targets, predictions)
        self.logs["performance"].append([epoch+1, auc, f1])
        return auc, f1

    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance"] = [["Epoch", "AUC", "F1"]]
        self.logs["training_time"] = [["Epoch", "Seconds"]]
