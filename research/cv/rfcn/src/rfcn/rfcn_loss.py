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
"""Rfcn Loss network."""

import numpy as np
import mindspore.numpy as np2

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor

class Loss(nn.Cell):
    """
    Rfcn Loss subnet.

    Args:
        config (dict) - Config.
        num_classes (int) - Class number.

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        Loss(config=config, num_classes = 81)
    """
    def __init__(self,
                 config,
                 num_classes
                 ):
        super(Loss, self).__init__()
        cfg = config
        self.dtype = np.float32
        self.ms_type = mstype.float32
        self.rfcn_loss_cls_weight = Tensor(np.array(cfg.rfcn_loss_cls_weight).astype(self.dtype))
        self.rfcn_loss_reg_weight = Tensor(np.array(cfg.rfcn_loss_reg_weight).astype(self.dtype))
        self.num_classes = num_classes
        self.logicaland = P.LogicalAnd()
        self.loss_cls = P.SoftmaxCrossEntropyWithLogits()
        self.loss_bbox = P.SmoothL1Loss(beta=1.0)
        self.onehot = P.OneHot()
        self.greater = P.Greater()
        self.cast = P.Cast()
        self.sum_loss = P.ReduceSum()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.value = Tensor(1.0, self.ms_type)


    def construct(self, x_cls, x_reg, bbox_targets, labels, mask):
        """rfcn loss construct"""
        if self.training:
            labels = self.onehot(labels, self.num_classes, self.on_value, self.off_value)
            loss, loss_cls, loss_reg, loss_print = self.loss(x_cls, x_reg, bbox_targets, labels, mask)
            out = (loss, loss_cls, loss_reg, loss_print)
        else:
            out = (x_cls, (x_cls / self.value), x_reg, x_cls)

        return out


    def loss(self, cls_score, bbox_pred, bbox_targets, labels, weights):
        """Loss method."""
        loss_print = ()
        loss_cls, _ = self.loss_cls(cls_score, labels)
        bbox_pred = bbox_pred[:, 4:8]
        loss_reg = self.loss_bbox(bbox_pred, bbox_targets)
        loss_loc = np2.sum(loss_reg, axis=1) / 4
        # compute total loss
        weights = self.cast(weights, self.ms_type)
        loss_cls = loss_cls * weights
        loss_loc = loss_loc * weights

        loss = loss_loc * self.rfcn_loss_reg_weight + loss_cls * self.rfcn_loss_cls_weight
        loss = np2.sum(loss) / self.sum_loss(weights, (0,))
        loss_print += (loss_cls, loss_loc)
        return loss, loss_cls, loss_reg, loss_print
