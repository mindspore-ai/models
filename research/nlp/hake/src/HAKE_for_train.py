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
"""HAKE for training"""

import mindspore
from mindspore import nn
from mindspore.ops import Softmax, stop_gradient, Log, Sigmoid, Squeeze, Cast, P, Depend, composite


class HAKENetworkWithLoss_HEAD(nn.Cell):
    """
    Model network loss definition for head negative training data
    Args:
        hake_network: base hake model
        adversarial_temperature: global hyper-parameters
    """

    def __init__(self, hake_network, adversarial_temperature):
        super().__init__()
        self.hake = hake_network

        self.softmax = Softmax(axis=1)
        self.log = Log()
        self.sigmoid = Sigmoid()
        self.squeeze = Squeeze(axis=1)
        self.adversarial_temperature = adversarial_temperature

    def construct(self, positive_sample, negative_sample, subsampling_weight):
        """ calculate the loss of head negative type"""
        subsampling_weight = self.squeeze(subsampling_weight)
        # negative scores
        negative_score = self.hake.construct_head((positive_sample, negative_sample))
        negative_score = (stop_gradient(self.softmax(negative_score * self.adversarial_temperature))
                          * self.log(self.sigmoid(-negative_score))).sum(axis=1)

        # positive scores
        positive_score = self.hake.construct_single(positive_sample)
        positive_score = self.squeeze(self.log(self.sigmoid(positive_score)))

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss


class HAKENetworkWithLoss_TAIL(nn.Cell):
    """
    Model network loss definition for tail negative training data
    Args:
        hake_network: base hake model
        adversarial_temperature: global hyper-parameters
    """

    def __init__(self, hake_network, adversarial_temperature):
        super().__init__()
        self.hake = hake_network
        self.adversarial_temperature = adversarial_temperature

        self.softmax = Softmax(axis=1)
        self.log = Log()
        self.sigmoid = Sigmoid()
        self.squeeze = Squeeze(axis=1)

    def construct(self, positive_sample, negative_sample, subsampling_weight):
        """ calculate the loss of tail negative type"""
        subsampling_weight = self.squeeze(subsampling_weight)

        # negative scores
        negative_score = self.hake.construct_tail((positive_sample, negative_sample))
        negative_score = (stop_gradient(self.softmax(negative_score * self.adversarial_temperature))
                          * self.log(self.sigmoid(-negative_score))).sum(axis=1)
        # positive scores
        positive_score = self.hake.construct_single(positive_sample)  # batch_type: BatchType.SINGLE
        positive_score = self.squeeze(self.log(self.sigmoid(positive_score)))

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss


class HAKETrainOneStepCell(nn.Cell):
    """
    Encapsulation class of transformer network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super().__init__()
        self.network = network
        self.network.set_train()
        self.weights = mindspore.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = composite.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.cast = Cast()
        self.depend = Depend()

    def construct(self, positive_sample, negative_sample, subsampling_weight):
        """Defines the computation performed."""

        weights = self.weights
        loss = self.network(positive_sample, negative_sample, subsampling_weight)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(positive_sample, negative_sample, subsampling_weight, sens)

        return self.depend(loss, self.optimizer(grads))
