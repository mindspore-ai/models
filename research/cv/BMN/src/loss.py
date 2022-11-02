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

import logging
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore import dtype as mstype
log = logging.getLogger(__name__)

def get_mask(tscale):
    mask = np.zeros([tscale, tscale], np.float32)
    for i in range(tscale):
        for j in range(i, tscale):
            mask[i, j] = 1
    return Tensor(mask)

class PEM_CLS_Loss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(PEM_CLS_Loss, self).__init__(reduction)
        self.reduce_sum = ops.ReduceSum()
        self.log = ops.Log()
        self.cast = ops.Cast()

    def construct(self, logits, labels, mask):
        pmask = self.cast(labels > 0.9, mstype.float32)
        nmask = self.cast(labels <= 0.9, mstype.float32)
        nmask = nmask * mask

        num_positive = self.reduce_sum(pmask)
        num_entries = num_positive + self.reduce_sum(nmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 1e-6
        loss_pos = coef_1 * self.log(logits + epsilon) * pmask
        loss_neg = coef_0 * self.log(1.0 - logits + epsilon) * nmask
        loss = -1 * self.reduce_sum(loss_pos + loss_neg) / num_entries
        return loss

class PEM_Reg_Loss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(PEM_Reg_Loss, self).__init__(reduction)
        self.reduce_sum = ops.ReduceSum()
        self.mse_loss = nn.MSELoss()
        self.logical_and = ops.LogicalAnd()
        self.cast = ops.Cast()
        self.uniform = ops.UniformReal()
        self.ones = ops.Ones()

    def construct(self, logits, labels, mask):
        u_hmask = self.cast(labels > 0.7, mstype.float32)
        u_mmask = self.cast(self.logical_and(labels <= 0.7, labels > 0.3), mstype.float32)
        u_lmask = self.cast(self.logical_and(labels <= 0.3, labels > 0.), mstype.float32)
        u_lmask = u_lmask * mask

        num_h = self.reduce_sum(u_hmask)
        num_m = self.reduce_sum(u_mmask)
        num_l = self.reduce_sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = self.uniform(labels.shape)
        u_smmask = u_mmask * u_smmask
        u_smmask = self.cast((u_smmask > (1. - r_m)), mstype.float32)

        r_l = num_h / num_l
        u_slmask = self.uniform(labels.shape)
        u_slmask = u_lmask * u_slmask
        u_slmask = self.cast((u_slmask > (1. - r_l)), mstype.float32)

        weights = u_hmask + u_smmask + u_slmask
        logits = self.cast(logits, mstype.float32)
        labels = self.cast(labels, mstype.float32)
        loss = self.mse_loss(logits * weights, labels * weights)
        loss = 0.5 * self.reduce_sum(loss * self.ones(weights.shape, mstype.float32)) / self.reduce_sum(weights)

        return loss

class Bi_Loss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(Bi_Loss, self).__init__(reduction)
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.log = ops.Log()

    def construct(self, logits, labels):
        pmask = self.cast(labels > 0.5, mstype.float32)
        num_entries = pmask.size
        num_positive = self.reduce_sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 1e-6
        loss_pos = coef_1 * self.log(logits + epsilon) * pmask
        loss_neg = coef_0 * self.log(1.0 - logits + epsilon) * (1.0 - pmask)
        loss = -1 * self.reduce_mean(loss_pos + loss_neg)
        return loss

class TEM_Loss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(TEM_Loss, self).__init__(reduction)
        self.bi_loss = Bi_Loss()

    def construct(self, pred_start, pred_end, gt_start, gt_end):

        loss_start = self.bi_loss(pred_start, gt_start)
        loss_end = self.bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

class BMN_Loss(nn.Cell):
    def __init__(self, bm_mask, mode='train'):
        super(BMN_Loss, self).__init__()
        self.pem_reg_loss = PEM_Reg_Loss()
        self.pem_cls_loss = PEM_CLS_Loss()
        self.tem_loss = TEM_Loss()
        self.stack = ops.Stack()
        self.unstack = ops.Unstack(axis=1)
        self.slice = ops.Slice()
        self.bm_mask = bm_mask
        self.mode = mode

    def construct(self, pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end):
        pred_bm_reg = pred_bm[:, 0]
        pred_bm_cls = pred_bm[:, 1]

        gt_iou_map = gt_iou_map * self.bm_mask

        pem_reg_loss = self.pem_reg_loss(pred_bm_reg, gt_iou_map, self.bm_mask)
        pem_cls_loss = self.pem_cls_loss(pred_bm_cls, gt_iou_map, self.bm_mask)
        tem_loss = self.tem_loss(pred_start, pred_end, gt_start, gt_end)

        loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
        if self.mode == "train":
            return loss
        losses = self.stack([loss, tem_loss, pem_reg_loss, pem_cls_loss])
        return losses
