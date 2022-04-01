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
from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C


class Loss(nn.Cell):
    """Softmax cross-entropy loss with masking."""
    def __init__(self, label, mask, weight_decay, param):
        super(Loss, self).__init__(auto_prefix=False)
        self.label = Tensor(label)
        self.mask = Tensor(mask)
        self.loss = P.SoftmaxCrossEntropyWithLogits()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)
        self.mean = P.ReduceMean()
        self.cast = P.Cast()
        self.l2_loss = P.L2Loss()
        self.reduce_sum = P.ReduceSum()
        self.weight_decay = weight_decay
        self.param = param

    def construct(self, preds):
        """Calculate loss"""
        param = self.l2_loss(self.param)
        loss = self.weight_decay * param
        preds = self.cast(preds, mstype.float32)
        loss = loss + self.loss(preds, self.label)[0]
        mask = self.cast(self.mask, mstype.float32)
        mask_reduce = self.mean(mask)
        mask = mask / mask_reduce
        loss = loss*self.mask
        loss = self.mean(loss)
        return loss



class Accuracy(nn.Cell):
    """Accuracy with masking."""
    def __init__(self, label, mask):
        super(Accuracy, self).__init__(auto_prefix=False)
        self.label = Tensor(label)
        self.mask = Tensor(mask)
        self.equal = P.Equal()
        self.argmax = P.Argmax()
        self.cast = P.Cast()
        self.mean = P.ReduceMean()

    def construct(self, preds):
        """Calculate accuracy"""
        preds = self.cast(preds, mstype.float32)
        correct_prediction = self.equal(self.argmax(preds), self.argmax(self.label))
        accuracy_all = self.cast(correct_prediction, mstype.float32)
        mask = self.cast(self.mask, mstype.float32)
        mask_reduce = self.mean(mask)
        mask = mask / mask_reduce
        accuracy_all *= mask
        return self.mean(accuracy_all)


class LossAccuracyWrapper(nn.Cell):
    """ Wraps the DGCN model with loss and accuracy cell"""
    def __init__(self, network, label, mask, weight_decay):
        super(LossAccuracyWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = Loss(label, mask, weight_decay, network.trainable_params()[0])
        self.accuracy = Accuracy(label, mask)

    def construct(self, adj, ppmi, feature, ret):
        diffpreds, ppmipreds = self.network(adj, ppmi, feature)
        diffloss = self.loss(diffpreds)
        ppmiloss = self.loss(ppmipreds)
        loss = diffloss + ret*((diffloss -ppmiloss)**2)
        accuracy = self.accuracy(diffpreds)
        return loss, accuracy


class LossWrapper(nn.Cell):
    """Wraps the GCN model with loss. """
    def __init__(self, network, label, mask, weight_decay):
        super(LossWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = Loss(label, mask, weight_decay, network.trainable_params()[0])

    def construct(self, adj, ppmi, feature, ret):
        diffpreds, ppmipreds = self.network(adj, ppmi, feature)
        diffloss = self.loss(diffpreds)
        ppmiloss = self.loss(ppmipreds)
        loss = diffloss + ret*((diffloss-ppmiloss)**2)
        return loss


class TrainOneStepCell(nn.Cell):
    """Network training package class."""
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, adj, ppmi, feature, ret):
        weights = self.weights
        loss = self.network(adj, ppmi, feature, ret)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(adj, ppmi, feature, ret, sens)
        self.optimizer(grads)
        return loss


class TrainNetWrapper(nn.Cell):
    """Wraps the GCN model with optimizer"""
    def __init__(self, network, label, mask, weight_decay, learning_rate):
        super(TrainNetWrapper, self).__init__(auto_prefix=False)
        self.network = network
        loss_net = LossWrapper(network, label, mask, weight_decay)
        optimizer = nn.Adam(loss_net.trainable_params(),
                            learning_rate=learning_rate)
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)
        self.accuracy = Accuracy(label, mask)

    def construct(self, adj, ppmi, feature, ret):
        loss = self.loss_train_net(adj, ppmi, feature, ret)
        accuracy = self.accuracy(self.network(adj, ppmi, feature)[0])
        return loss, accuracy
        