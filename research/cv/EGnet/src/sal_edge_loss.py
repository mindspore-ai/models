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

"""SalEdgeLoss define"""

import mindspore as ms
from mindspore import nn
from mindspore.ops import Equal, Cast, ReduceSum, BCEWithLogitsLoss, OnesLike
from mindspore import Parameter


class SalEdgeLoss(nn.Cell):
    """
    salience and edge loss
    """
    def __init__(self, n_ave_grad, batch_size):
        super(SalEdgeLoss, self).__init__()
        self.n_ave_grad = n_ave_grad
        self.batch_size = batch_size
        self.sum = ReduceSum()
        self.equal = Equal()
        self.cast = Cast()
        self.ones = OnesLike()
        self.bce = BCEWithLogitsLoss(reduction="sum")
        self.zero = ms.Tensor(0, dtype=ms.float32)
        # for log
        self.sal_loss = Parameter(default_input=0.0, requires_grad=False)
        self.edge_loss = Parameter(default_input=0.0, requires_grad=False)
        self.total_loss = Parameter(default_input=0.0, requires_grad=False)

    def bce2d_new(self, predict, target):
        """
        binary cross entropy loss with weights
        """
        pos = self.cast(self.equal(target, 1), ms.float32)
        neg = self.cast(self.equal(target, 0), ms.float32)

        num_pos = self.sum(pos)
        num_neg = self.sum(neg)
        num_total = num_pos + num_neg

        alpha = num_neg / num_total
        beta = 1.1 * num_pos / num_total
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = alpha * pos + beta * neg
        bce = BCEWithLogitsLoss(reduction="sum")
        return bce(predict, target, weights, self.ones(predict))

    def construct(self, up_edge, up_sal, up_sal_f, sal_label, sal_edge):
        """
        compute loss
        """
        edge_loss = self.zero
        for ix in up_edge:
            edge_loss += self.bce2d_new(ix, sal_edge)
        edge_loss = edge_loss / (self.n_ave_grad * self.batch_size)

        sal_loss1 = self.zero
        sal_loss2 = self.zero
        for ix in up_sal:
            bce = BCEWithLogitsLoss(reduction="sum")
            sal_loss1 += bce(ix, sal_label, self.ones(ix), self.ones(ix))
        for ix in up_sal_f:
            bce = BCEWithLogitsLoss(reduction="sum")
            sal_loss2 += bce(ix, sal_label, self.ones(ix), self.ones(ix))

        sal_loss = (sal_loss1 + sal_loss2) / (self.n_ave_grad * self.batch_size)
        loss = sal_loss + edge_loss
        self.sal_loss, self.edge_loss, self.total_loss = sal_loss, edge_loss, loss
        return loss


class WithLossCell(nn.Cell):
    """
    loss cell
    """
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, sal_label, sal_edge):
        """
        compute loss
        """
        up_edge, up_sal, up_sal_f = self.backbone(data)
        return self.loss_fn(up_edge, up_sal, up_sal_f, sal_label, sal_edge)

    @property
    def backbone_network(self):
        return self.backbone
