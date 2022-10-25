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
"""SGCN runner."""
import time

import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore import save_checkpoint
from mindspore.common.initializer import XavierUniform
from mindspore.common.initializer import Zero
from sklearn.model_selection import train_test_split

from src.metrics import TrainNetWrapper
from src.ms_utils import calculate_auc
from src.ms_utils import setup_features
from src.signedsageconvolution import SignedSAGEConvolutionBase
from src.signedsageconvolution import SignedSAGEConvolutionDeep


class SignedGraphConvolutionalNetwork(nn.Cell):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354

    SGCN Initialization.
    """
    def __init__(self, x, norm, norm_embed, bias):
        super().__init__()
        self.X = Tensor(x, mstype.float32)
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
        neurons = (96, 64, 32)
        self.positive_base_aggregator = SignedSAGEConvolutionBase(
            self.X.shape[1] * 2,
            neurons[0],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.negative_base_aggregator = SignedSAGEConvolutionBase(
            self.X.shape[1] * 2,
            neurons[0],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.positive_aggregator_1 = SignedSAGEConvolutionDeep(
            3 * neurons[0],
            neurons[1],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.positive_aggregator_2 = SignedSAGEConvolutionDeep(
            3 * neurons[1],
            neurons[2],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.negative_aggregator_1 = SignedSAGEConvolutionDeep(
            3 * neurons[0],
            neurons[1],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.negative_aggregator_2 = SignedSAGEConvolutionDeep(
            3 * neurons[1],
            neurons[2],
            norm=self.norm,
            norm_embed=self.norm_embed,
            bias=self.bias
        )
        self.regression_weights = Parameter(
            Tensor(shape=(4 * neurons[-1], 3), dtype=mstype.float32, init=XavierUniform(gain=1.0))
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
        h_pos_1 = self.tanh(self.positive_base_aggregator(self.X, removed_pos))
        h_neg_1 = self.tanh(self.negative_base_aggregator(self.X, removed_neg))

        h_pos_2 = self.tanh(self.positive_aggregator_1(h_pos_1, h_neg_1, removed_pos, removed_neg))
        h_neg_2 = self.tanh(self.negative_aggregator_1(h_neg_1, h_pos_1, removed_pos, removed_neg))

        h_pos_3 = self.tanh(self.positive_aggregator_2(h_pos_2, h_neg_2, removed_pos, removed_neg))
        h_neg_3 = self.tanh(self.negative_aggregator_2(h_neg_2, h_pos_2, removed_pos, removed_neg))

        z = self.op((h_pos_3, h_neg_3))
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

    def setup_dataset(self):
        """
        Returns:
            self.X(np.ndarray): Dataset.
            self.positive_edges(Tensor): Positive edges for training.
            self.negative_edges(Tensor): Negative edges for training.
            self.test_positive_edges(Tensor): Positive edges for testing.
            self.test_negative_edges(Tensor): Negative edges for testing.
        """
        positive_edges, test_positive_edges = train_test_split(
            self.edges["positive_edges"],
            test_size=self.args.test_size,
            random_state=1,
        )

        negative_edges, test_negative_edges = train_test_split(
            self.edges["negative_edges"],
            test_size=self.args.test_size,
            random_state=1,
        )

        self.test_positive_edges = np.array(test_positive_edges)
        self.test_negative_edges = np.array(test_negative_edges)
        self.targets = [0] * len(self.test_positive_edges) + [1] * len(self.test_negative_edges)
        self.score_edges = Tensor(
            np.concatenate([self.test_positive_edges.reshape(-1), self.test_negative_edges.reshape(-1)]),
            mstype.int32,
        )

        self.ecount = len(positive_edges) + len(negative_edges)

        self.X = setup_features(self.args, positive_edges, negative_edges, self.edges["ncount"])
        self.positive_edges = np.array(positive_edges, dtype=np.int32).T
        self.negative_edges = np.array(negative_edges, dtype=np.int32).T
        self.y = np.array([0 if i < int(self.ecount / 2) else 1 for i in range(self.ecount)] + [2] * (self.ecount * 2))
        self.y = np.array(self.y, np.int32)
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
        self.removed_pos = Tensor(self.remove_self_loops(self.positive_edges))
        self.removed_neg = Tensor(self.remove_self_loops(self.negative_edges))
        num_nodes = self.X.shape[0]

        self.optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.epochs = self.args.epochs
        train_net = TrainNetWrapper(self.model, self.y, weight_decay=self.args.weight_decay,
                                    learning_rate=self.args.learning_rate, lamb=self.args.lamb)
        best_auc, best_f1 = 0, 0
        train_net.set_train()
        t0 = time.time()
        for epoch in range(self.epochs):
            t = time.time()
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

    @staticmethod
    def structured_sampling(edge_index, num_nodes=None):
        """
        Samples a negative edge for every positive edge
        Args:
            edge_index (np.ndarray): The edge indices.
            num_nodes(Int): Number of nodes.

        Returns:
            i(Int): rows of edge indices.
            j(Int): columns of edge indices.
            k(Int): A tensor with given values.
        """
        i, j = edge_index
        idx_1 = i * num_nodes + j
        k = np.random.uniform(0, num_nodes, i.shape[0]).round().astype(np.int32)
        idx_2 = i * num_nodes + k
        mask = np.isin(idx_2, idx_1).squeeze()
        rest = np.squeeze(mask.nonzero(), axis=0).reshape(-1)
        while rest.size > 0:
            tmp = np.random.uniform(0, num_nodes, rest.size).round().astype(np.int32)
            idx_3 = i[rest] * num_nodes + tmp
            mask = np.isin(idx_3, idx_1).squeeze()
            k[rest] = tmp
            rest_index = np.squeeze(mask.nonzero(), axis=0).reshape(-1)
            rest = rest[rest_index]

        i_tensor = Tensor.from_numpy(np.ascontiguousarray(i))
        j_tensor = Tensor.from_numpy(np.ascontiguousarray(j))
        k_tensor = Tensor.from_numpy(np.ascontiguousarray(k))
        return i_tensor, j_tensor, k_tensor

    @staticmethod
    def remove_self_loops(edge_index):
        """
        remove self loops
        Args:
            edge_index (np.ndarray): The edge indices.

        Returns:
            np.ndarray: removed self loops
        """
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        return edge_index

    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        Args:
            epoch (LongTensor): Training epochs.

        Returns:
            auc(Float32): AUC result.
            f1(Float32): F1-Score result.
        """
        train_z = self.model(self.removed_pos, self.removed_neg)

        scores = ops.matmul(
            train_z[self.score_edges].reshape(-1, train_z.shape[1] * 2),
            self.model.regression_weights,
        ) + self.model.regression_bias
        probability_scores = ops.Exp()(ops.Softmax(axis=1)(scores))
        predictions = probability_scores[:, 0] / probability_scores[:, 0: 2].sum(1)
        predictions = predictions.asnumpy()
        auc, f1 = calculate_auc(self.targets, predictions)
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
