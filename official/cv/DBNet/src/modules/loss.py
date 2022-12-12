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
# This file refers to the project https://github.com/MhLiao/DB.git
"""Loss functions."""

from mindspore import nn, ops
import mindspore as ms
import mindspore.numpy as mnp


class L1BalanceCELoss(nn.LossBase):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5, bce_replace="bceloss"):
        super(L1BalanceCELoss, self).__init__()

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()

        if bce_replace == "bceloss":
            self.bce_loss = BalanceCrossEntropyLoss()
        elif bce_replace == "diceloss":
            self.bce_loss = DiceLoss()
        else:
            raise ValueError(f"bce_replace should be in ['bceloss', 'diceloss'], but get {bce_replace}")

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def construct(self, pred, gt, gt_mask, thresh_map, thresh_mask):
        """
        pred: A dict which contains predictions.
            thresh: The threshold prediction
            binary: The text segmentation prediction.
            thresh_binary: Value produced by `step_function(binary - thresh)`.
        gt: Text regions bitmap gt.
        mask: Ignore mask, pexels where value is 1 indicates no contribution to loss.
        thresh_mask: Mask indicates regions cared by thresh supervision.
        thresh_map: Threshold gt.
        """
        bce_loss_output = self.bce_loss(pred['binary'], gt, gt_mask)

        if 'thresh' in pred:
            l1_loss = self.l1_loss(pred['thresh'], thresh_map, thresh_mask)
            dice_loss = self.dice_loss(pred['thresh_binary'], gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss_output * self.bce_scale
        else:
            loss = bce_loss_output

        return loss


class DiceLoss(nn.LossBase):

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def construct(self, pred, gt, mask, weights=None):
        """
        pred: one or two heatmaps of shape (N, 1, H, W),
              the losses of two heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        """
        pred = pred.squeeze(axis=1)
        gt = gt.squeeze(axis=1)
        if weights is not None:
            mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union

        return loss


class MaskL1Loss(nn.LossBase):
    """Mask L1 loss."""
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def construct(self, pred, gt, mask):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        pred = pred.squeeze(axis=1)
        mask_sum = mask.sum()
        return ((pred - gt).abs() * mask).sum() / (mask_sum + self.eps)


class BalanceCrossEntropyLoss(nn.LossBase):
    """Balanced cross entropy loss."""
    def __init__(self, negative_ratio=3, eps=1e-6):

        super(BalanceCrossEntropyLoss, self).__init__()

        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bceloss = nn.BCELoss(reduction="none")
        # self.sort = ops.TopK()
        self.sort = ops.Sort(descending=False)
        self.min = ops.Minimum()
        self.cast = ops.Cast()
        self.gather = ops.GatherNd()
        self.stack = ops.Stack(axis=1)
        self.unsqueeze = ops.ExpandDims()

    def construct(self, pred, gt, mask):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """

        # see this example for workaround of hard negative mining:
        # https://gitee.com/zhao_ting_v/ssd_benchmark/blob/master/src/ssd_benchmark.py

        pred = pred.squeeze(axis=1)
        gt = gt.squeeze(axis=1)
        pos = (gt * mask).astype(ms.float32)
        neg = ((1 - gt) * mask).astype(ms.float32)

        positive_count = pos.sum(axis=(1, 2), keepdims=True).astype(ms.int32)
        negative_count = neg.sum(axis=(1, 2), keepdims=True).astype(ms.int32)

        negative_count = self.min(negative_count, positive_count * self.negative_ratio).squeeze(axis=(1, 2))

        loss = self.bceloss(pred.astype(ms.float32), gt.astype(ms.float32))

        positive_loss = loss * pos
        N = loss.shape[0]
        negative_loss = (loss * neg).view(N, -1)

        negative_value, _ = self.sort(negative_loss)
        batch_iter = mnp.arange(N)
        neg_index = self.stack((batch_iter, negative_count))
        min_neg_score = self.unsqueeze(self.gather(negative_value, neg_index), 1)

        masked_neg_loss = self.cast(negative_loss >= min_neg_score, ms.float32) # filter out losses less than topk loss.

        masked_neg_loss = ops.stop_gradient(masked_neg_loss)

        masked_neg_loss = masked_neg_loss * negative_loss

        balance_loss = (positive_loss.sum() + masked_neg_loss.sum()) / \
                       ((positive_count + negative_count).astype(ms.float32).sum() + self.eps)

        return balance_loss
