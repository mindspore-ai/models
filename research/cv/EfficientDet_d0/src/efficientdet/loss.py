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
""" loss definition """
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore import numpy as np
import mindspore.ops as op
from mindspore import Tensor
from src.efficientdet.utils import BoxEncoder


class Maximum(nn.Cell):
    """ maximum op """
    def __init__(self):
        super(Maximum, self).__init__()
        self.max = op.Maximum()
        self.tile_op = op.Tile()
        self.expand_dims = op.ExpandDims()
        self.squeeze = op.Squeeze()

    def construct(self, a, b):
        """ forward """
        a = self.expand_dims(self.squeeze(a), 1)
        a = self.tile_op(a, (1, 128))

        b = self.expand_dims(self.squeeze(b), 0)
        b = self.tile_op(b, (49104, 1))

        return self.max(a, b)


class Minimum(nn.Cell):
    """ minimum op"""
    def __init__(self):
        super(Minimum, self).__init__()
        self.min = op.Minimum()
        self.tile_op = op.Tile()
        self.expand_dims = op.ExpandDims()
        self.squeeze = op.Squeeze()

    def construct(self, a, b):
        """ forward """
        a = self.expand_dims(self.squeeze(a), 1)
        a = self.tile_op(a, (1, 128))
        b = self.expand_dims(self.squeeze(b), 0)
        b = self.tile_op(b, (49104, 1))
        return self.min(a, b)


class IOU(nn.Cell):
    """ iou """
    def __init__(self):
        super(IOU, self).__init__()
        self.expand_dims = op.ExpandDims()
        self.squeeze = op.Squeeze()
        self.max = Maximum()
        self.min = Minimum()

        self.min_value0 = Tensor(0, mindspore.float32)
        self.min_value1 = Tensor(1e-8, mindspore.float32)
        self.max_value = Tensor(10e6, mindspore.float32)

    def construct(self, a, b):
        """ iou """
        # a(anchor) [boxes, (y1, x1, y2, x2)]
        # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = self.min(a[:, 3], b[:, 2]) - self.max(a[:, 1], b[:, 0])
        ih = self.min(a[:, 2], b[:, 3]) - self.max(a[:, 0], b[:, 1])
        iw = op.clip_by_value(iw, clip_value_min=self.min_value0, clip_value_max=self.max_value)
        ih = op.clip_by_value(ih, clip_value_min=self.min_value0, clip_value_max=self.max_value)

        ua = self.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1) + area - iw * ih
        ua = op.clip_by_value(ua, clip_value_min=self.min_value1, clip_value_max=self.max_value)

        intersection = iw * ih
        IoU = intersection / ua

        return IoU


class FocalLoss(nn.Cell):
    """ focal loss for efficientdet"""

    def __init__(self):
        super(FocalLoss, self).__init__()

        self.pow = op.Pow()
        self.log = op.Log()
        self.maximum = op.Maximum()
        self.less = op.Less()
        self.greater = op.Greater()
        self.abs = op.Abs()
        self.select = op.Select()
        self.min_value1 = Tensor(1e-4, mindspore.float32)
        self.max_value1 = Tensor(1.0 - 1e-4, mindspore.float32)
        self.tile = P.Tile()
        self.expand_dims = op.ExpandDims()
        self.box_encoder = BoxEncoder()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.argmaxwithvalue = op.ArgMaxWithValue(axis=2, keep_dims=True)
        self.iou = IOU()
        self.cast = op.Cast()
        self.stack = op.Stack()
        self.gatherD = op.GatherD()
        self.onehot = op.OneHot()
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.argmax = op.Argmax(1)
        self.reduce_mean = op.ReduceMean(keep_dims=False)
        self.oneslike = op.OnesLike()
        self.squeeze = op.Squeeze()
        self.squeeze_dim2 = op.Squeeze(axis=2)
        self.bitwise_or = op.BitwiseOr()
        self.bitwise_and = op.BitwiseAnd()
        self.concat = op.Concat(2)
        self.inf = np.inf
        self.min = Tensor(1.0, mindspore.float32)
        self.max = Tensor(49104, mindspore.float32)
        self.squeeze_dim1 = op.Squeeze(axis=1)
        self.not_op = op.LogicalNot()
        self.const_1 = Tensor(1, mstype.float32)
        self.const_512 = Tensor(512, mstype.float32)

    def construct(self, regressions, classifications, anchor, annotations):
        """ loss calc """

        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classifications = op.clip_by_value(classifications, self.min_value1, self.max_value1)

        gt_loc = annotations[:, :, 0:4:1]
        gt_label = self.squeeze_dim2(annotations[:, :, 4:5:1])

        anchor_widths = anchor[:, :, 3] - anchor[:, :, 1]
        anchor_heights = anchor[:, :, 2] - anchor[:, :, 0]
        anchor_ctr_x = anchor[:, :, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, :, 0] + 0.5 * anchor_heights

        IoU = ()
        for i in range(batch_size):
            gt_loc_i = self.squeeze(self.cast(gt_loc[i:i + 1:1, ::], mindspore.float32))
            anchor_i = self.squeeze(self.cast(anchor[i:i + 1:1, ::], mindspore.float32))
            iou = self.iou(anchor_i, gt_loc_i)
            IoU += (iou,)

        IoU = self.stack(IoU)
        IoU_argmax, IoU_max = self.argmaxwithvalue(IoU)

        IoU_argmax = self.squeeze_dim2(IoU_argmax)
        gt_label_int = self.cast(gt_label, mindspore.int32)
        want_label = self.gatherD(gt_label_int, 1, IoU_argmax)

        target = self.cast(self.onehot(want_label, classifications.shape[-1], self.on_value,
                                       self.off_value), mindspore.int32)

        mask_greater = F.cast(self.not_op(self.less(IoU_max, 0.5)), mindspore.int32)
        mask_greater_2 = self.cast(self.squeeze_dim2(mask_greater), mindspore.float32)

        positive_num = self.squeeze_dim1(self.reduce_sum(mask_greater_2, 1))
        positive_num = op.clip_by_value(positive_num, self.min, self.max)

        mask_greater = self.tile(mask_greater, (1, 1, classifications.shape[-1]))

        target_masked = self.bitwise_and(mask_greater, target)

        target_masked_bool = F.cast(target_masked, mindspore.bool_)

        mask_less = F.cast(self.less(IoU_max, 0.4), mstype.int32)

        mask_less = self.tile(mask_less, (1, 1, classifications.shape[-1]))

        mask_useful = self.bitwise_or(mask_less, mask_greater)

        alpha = self.oneslike(target) * alpha

        alpha_factor = self.select(target_masked_bool, alpha, 1. - alpha)
        focal_weight = self.select(target_masked_bool, 1. - classifications, classifications)

        focal_weight = alpha_factor * self.pow(focal_weight, gamma)

        target = target * mask_greater

        bce = -(target * self.log(classifications) + (1.0 - target) * self.log(
            1.0 - classifications))

        cls_loss_total = focal_weight * bce * mask_useful

        cls_loss_total = self.reduce_sum(cls_loss_total, (1, 2)) / positive_num

        cls_loss = self.reduce_mean(cls_loss_total, ())

        # ################################ regression loss ##############################

        IoU_argmax_ex = self.expand_dims(IoU_argmax, -1)
        IoU_argmax_loc = self.tile(IoU_argmax_ex, (1, 1, 4))
        target_loc = self.gatherD(gt_loc, 1, IoU_argmax_loc)
        mask_greater_loc = F.cast(self.greater(IoU_max, 0.5), mindspore.int32)
        mask_greater_loc = self.tile(mask_greater_loc, (1, 1, 4))

        gt_loc = op.clip_by_value(self.box_encoder(F.cast(target_loc, mindspore.float32)), self.const_1, self.const_512)

        gt_ctr_x = gt_loc[:, :, 0]
        gt_ctr_y = gt_loc[:, :, 1]
        gt_widths = gt_loc[:, :, 2]
        gt_heights = gt_loc[:, :, 3]

        targets_dx = self.expand_dims((gt_ctr_x - anchor_ctr_x) / anchor_widths, 2)
        targets_dy = self.expand_dims((gt_ctr_y - anchor_ctr_y) / anchor_heights, 2)
        targets_dw = self.expand_dims(self.log(gt_widths / anchor_widths), 2)
        targets_dh = self.expand_dims(self.log(gt_heights / anchor_heights), 2)

        targets = self.concat((targets_dy, targets_dx, targets_dh, targets_dw))

        regression_diff = self.abs(targets - regressions)
        regression_loss_total = self.select(
            self.less(regression_diff, 1.0 / 9.0),
            0.5 * 9.0 * self.pow(regression_diff, 2),
            regression_diff - 0.5 / 9.0
        ) * mask_greater_loc    # smooth l1 loss

        regression_loss_one = self.reduce_sum(regression_loss_total, (1, 2)) / 4
        regression_loss_one = regression_loss_one / positive_num

        regression_loss = self.reduce_mean(regression_loss_one, ())

        return cls_loss, regression_loss * 50
