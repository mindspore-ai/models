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
""" trainingcell.py """

import mindspore as ms
import mindspore.ops as P
import mindspore.numpy as msnp
from mindspore import Parameter, Tensor, ParameterTuple
from mindspore import nn


class CriterionWithNet(nn.Cell):
    """
    Mindpore module includes network and loss function
    """

    def __init__(self, backbone, ce_loss, tri_loss, kl_div, t=1, loss_func='id'):
        super(CriterionWithNet, self).__init__()
        self._backbone = backbone
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss
        self._kl_loss = kl_div
        self.t = t
        self.softmax = nn.Softmax()
        self.loss_func = loss_func
        self.loss_id = Parameter(Tensor([0.0], ms.float32), name="loss_id")
        self.loss_tri = Parameter(Tensor([0.0], ms.float32), name="loss_tri")
        self.acc = Parameter(Tensor([0.0], ms.float32), name="acc")

        self.cat = P.Concat(0)
        self.cast = P.Cast()
        self.sum = P.ReduceSum()
        self.max = P.ArgMaxWithValue(axis=1)
        self.eq = P.Equal()

    def get_acc(self, logits, label):
        predict, _ = self.max(logits)
        correct = self.eq(predict, label)
        return P.Div()(msnp.where(correct, 1.0, 0.0).sum(), label.shape[0])

    def construct(self, img1, img2, label1, label2):
        """
        v_observation[0] is logits， v_observation[1] is feature
        """
        imgs = self.cat((img1, img2))
        label = self.cat((label1, label2))
        label_ = self.cast(label, ms.int32)
        v_observation, v_representation, v_ms_observation, v_ms_representation, \
        i_observation, i_representation, i_ms_observation, i_ms_representation = \
            self._backbone(imgs)

        loss_id = 0.5 * (self._ce_loss(v_observation[1], label_) +
                         self._ce_loss(v_representation[1], label_)) + \
                  0.5 * (self._ce_loss(i_observation[1], label_) +
                         self._ce_loss(i_representation[1], label_)) + \
                  0.25 * (self._ce_loss(v_ms_observation[1], label_) +
                          self._ce_loss(v_ms_representation[1], label_)) + \
                  0.25 * (self._ce_loss(i_ms_observation[1], label_) +
                          self._ce_loss(i_ms_representation[1], label_))

        loss_total = 0

        if self.loss_func == "id":
            loss_total = loss_id
        elif self.loss_func == "id+tri":
            loss_tri = 0.5 * (self._tri_loss(v_observation[0], label) +
                              self._tri_loss(v_representation[0], label)) \
                       + 0.5 * (self._tri_loss(i_observation[0], label) +
                                self._tri_loss(i_representation[0], label)) \
                       + 0.25 * (self._tri_loss(v_ms_observation[0], label) +
                                 self._tri_loss(v_ms_representation[0], label)) \
                       + 0.25 * (self._tri_loss(i_ms_observation[0], label) +
                                 self._tri_loss(i_ms_representation[0], label))

            loss_total = loss_id + loss_tri

            P.Depend()(loss_tri, P.Assign()(self.loss_tri, loss_tri))

        acc_tmp = \
            self.get_acc(v_observation[1], label_) + self.get_acc(v_representation[1], label_) \
            + self.get_acc(i_observation[1], label_) + self.get_acc(i_representation[1], label_) \
            + self.get_acc(v_ms_observation[1], label_) + self.get_acc(v_ms_representation[1], label_) \
            + self.get_acc(i_ms_observation[1], label_) + self.get_acc(i_ms_representation[1], label_)

        P.Depend()(acc_tmp, P.Assign()(self.acc, acc_tmp / 8.0))
        P.Depend()(loss_id, P.Assign()(self.loss_id, loss_id))

        return loss_total

    @property
    def backbone_network(self):
        """
        return backbone
        """
        return self._backbone


class OptimizerWithNetAndCriterion(nn.Cell):
    """
    Mindspore Cell incldude Network, Optimizer and loss function.
    """

    def __init__(self, network, optimizer):
        super(OptimizerWithNetAndCriterion, self).__init__(auto_prefix=True)
        self.network = network
        self.weights = ParameterTuple(optimizer.parameters)
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True)

    def construct(self, img1, img2, label1, label2):
        weights = self.weights
        loss = self.network(img1, img2, label1, label2)
        grads = self.grad(self.network, weights)(img1, img2, label1, label2)
        P.Depend()(loss, self.optimizer(grads))
        return loss
