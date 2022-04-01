# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Loss and accuracy."""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.common.parameter import ParameterTuple
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context


class SGCNLoss(nn.Cell):
    """
    Calculates loss.
    """
    def __init__(self, regression_weights, regression_bias, lamb):
        super(SGCNLoss, self).__init__(auto_prefix=False)
        self.regression_weights, self.regression_bias = regression_weights, regression_bias
        self.print = ops.Print()
        self.reshape = ops.Reshape()
        self.size = ops.Size()
        self.pow = ops.Pow()
        self.concat = ops.Concat(axis=1)
        self.concat0 = ops.Concat(axis=0)
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.lamb = lamb

    def calculate_regression_loss(
            self, z, target,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k
    ):
        """
        Calculating the regression loss for all pairs of nodes.
        Args:
            z(Tensor): Hidden vertex representations.
            target(Tensor): Ground truth.
            regression_positive_i(Int): rows of positive edge indices.
            regression_positive_j(Int): columns of positive edge indices.
            regression_positive_k(Int): A tensor with given values.
            regression_negative_i(Int): rows of negative edge indices.
            regression_negative_j(Int): columns of negative edge indices.
            regression_negative_k(Int): A tensor with given values.

        Returns:
            loss_term(Tensor): regression_loss
        """
        i, j, k = regression_negative_i, regression_negative_j, regression_negative_k
        negative_z_i = z[i]
        negative_z_j = z[j]
        negative_z_k = z[k]
        i, j, k = regression_positive_i, regression_positive_j, regression_positive_k
        positive_z_i = z[i]
        positive_z_j = z[j]
        positive_z_k = z[k]
        pos = self.concat((positive_z_i, positive_z_j))
        neg = self.concat((negative_z_i, negative_z_j))
        surr_neg_i = self.concat((negative_z_i, negative_z_k))
        surr_neg_j = self.concat((negative_z_j, negative_z_k))
        surr_pos_i = self.concat((positive_z_i, positive_z_k))
        surr_pos_j = self.concat((positive_z_j, positive_z_k))
        features = self.concat0((pos, neg, surr_neg_i, surr_neg_j, surr_pos_i, surr_pos_j))
        predictions = ops.matmul(features, self.regression_weights) + self.regression_bias
        loss_term = self.ce(predictions, target).mean()
        return loss_term

    def calculate_positive_embedding_loss(self, z, positive_i, positive_j, positive_k):
        """
        Calculating the loss on the positive edge embedding distances
        Args:
            z(Tensor): Hidden vertex representations.
            positive_i(Int): rows of positive edge indices.
            positive_j(Int): columns of positive edge indices.
            positive_k(Int): A tensor with given values.

        Returns:
            ops.maximum(out, 0).mean(): positive_embedding_loss
        """
        i, j, k = positive_i, positive_j, positive_k
        out = self.pow((z[i] - z[j]), 2.0).sum(axis=1) - self.pow((z[i] - z[k]), 2.0).sum(axis=1)
        return ops.maximum(out, 0).mean()

    def calculate_negative_embedding_loss(self, z, negative_i, negative_j, negative_k):
        """
        Calculating the loss on the negative edge embedding distances
        Args:
            z(Tensor): Hidden vertex representations.
            negative_i(Int): rows of negative edge indices.
            negative_j(Int): columns of negative edge indices.
            negative_k(Int): A tensor with given values.

        Returns:
            ops.maximum(out, 0).mean(): negative_embedding_loss
        """
        i, j, k = negative_i, negative_j, negative_k
        out = self.pow((z[i] - z[k]), 2.0).sum(axis=1) - self.pow((z[i] - z[j]), 2.0).sum(axis=1)
        return ops.maximum(out, 0).mean()

    def construct(
            self, z, target,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k,
            positive_i, positive_j, positive_k,
            negative_i, negative_j, negative_k
    ):
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        Args:
            z(Tensor): Hidden vertex representations.
            target(Tensor): Ground truth.
            regression_positive_i(Int): rows of positive edge indices.
            regression_positive_j(Int): columns of positive edge indices.
            regression_positive_k(Int): A tensor with given values.
            regression_negative_i(Int): rows of negative edge indices.
            regression_negative_j(Int): columns of negative edge indices.
            regression_negative_k(Int): A tensor with given values.
            positive_i(Int): rows of positive edge indices.
            positive_j(Int): columns of positive edge indices.
            positive_k(Int): A tensor with given values.
            negative_i(Int): rows of negative edge indices.
            negative_j(Int): columns of negative edge indices.
            negative_k(Int): A tensor with given values.

        Returns:
            loss_term(Tensor): sgcn_loss
        """
        loss_term_1 = self.calculate_positive_embedding_loss(z, positive_i, positive_j, positive_k)
        loss_term_2 = self.calculate_negative_embedding_loss(z, negative_i, negative_j, negative_k)
        regression_loss = self.calculate_regression_loss(
            z, target,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k
        )
        loss_term = regression_loss + self.lamb * (loss_term_1 + loss_term_2)
        return loss_term


class LossWrapper(nn.Cell):
    """
    Wraps the GCN model with loss.
    """
    def __init__(self, network, label, lamb):
        super(LossWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.label = label
        self.loss = SGCNLoss(network.regression_weights, network.regression_bias, lamb)

    def construct(
            self, removed_pos, removed_neg,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k,
            positive_i, positive_j, positive_k,
            negative_i, negative_j, negative_k
    ):
        """Build a forward graph"""
        preds = self.network(removed_pos, removed_neg)
        loss = self.loss(
            preds, self.label,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k,
            positive_i, positive_j, positive_k,
            negative_i, negative_j, negative_k
        )
        return loss


class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell be trained without inputs.
    Backward graph will be created in the construct function to do parameter updating. Different
    parallel modes are available to run the training.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad(requires_grad=True)
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True, get_all=False)
        self.sens = sens
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.print = ops.Print()
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(
            self, removed_pos, removed_neg,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k,
            positive_i, positive_j, positive_k,
            negative_i, negative_j, negative_k
    ):
        """
        trainonestepcell construct
        Args:
            removed_pos(Tensor): Positive edges without self loops.
            removed_neg(Tensor): Negative edges without self loops.
            regression_positive_i(Int): rows of positive edge indices.
            regression_positive_j(Int): columns of positive edge indices.
            regression_positive_k(Int): A tensor with given values.
            regression_negative_i(Int): rows of negative edge indices.
            regression_negative_j(Int): columns of negative edge indices.
            regression_negative_k(Int): A tensor with given values.
            positive_i(Int): rows of positive edge indices.
            positive_j(Int): columns of positive edge indices.
            positive_k(Int): A tensor with given values.
            negative_i(Int): rows of negative edge indices.
            negative_j(Int): columns of negative edge indices.
            negative_k(Int): A tensor with given values.

        Returns:
            loss(Tensor): sgcn_loss
        """
        weights = self.weights
        loss = self.network(
            removed_pos, removed_neg,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k,
            positive_i, positive_j, positive_k,
            negative_i, negative_j, negative_k
        )
        sens = self.fill(self.dtype(loss), self.shape(loss), self.sens)
        grads = self.grad(self.network, weights)(
            removed_pos, removed_neg,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k,
            positive_i, positive_j, positive_k,
            negative_i, negative_j, negative_k,
            sens
        )
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


class TrainNetWrapper(nn.Cell):
    """
    Wraps the SGCN model with optimizer.
    """

    def __init__(self, network, label, weight_decay, learning_rate, lamb):
        super(TrainNetWrapper, self).__init__(auto_prefix=False)
        self.network = network
        loss_net = LossWrapper(network, label, lamb)
        optimizer = nn.Adam(
            loss_net.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)

    def construct(
            self, removed_pos, removed_neg,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k,
            positive_i, positive_j, positive_k,
            negative_i, negative_j, negative_k
    ):
        """Build a forward graph"""
        loss = self.loss_train_net(
            removed_pos, removed_neg,
            regression_positive_i, regression_positive_j, regression_positive_k,
            regression_negative_i, regression_negative_j, regression_negative_k,
            positive_i, positive_j, positive_k,
            negative_i, negative_j, negative_k
        )
        return loss
