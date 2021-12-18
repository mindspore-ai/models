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
"""DGCNet(res101) CE-loss."""
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore
from mindspore.ops import operations as P

from src.DualGCNNet import DualSeg_res101

def masked_fill(x, mask, value):
    mul = mindspore.ops.Mul()
    mask_o = ~mask
    x = mul(x, mask_o) + mul(value, mask)
    return x


class SoftmaxCrossEntropyLoss(nn.Cell):
    """SoftmaxCrossEntropyLoss"""
    def __init__(self, num_cls=21, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """SoftmaxCrossEntropyLoss.construct"""
        labels_int = self.cast(labels, mstype.int64)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float64)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


class OhemCrossEntropy2dTensor(nn.Cell):
    """OhemCrossEntropy2dTensor"""
    def __init__(self, ignore_label, ncls, thresh=0.6, min_kept=256, down_ratio=1):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_label = ignore_label
        self.num_cls = ncls
        self.thresh = thresh
        self.min_kept = min_kept
        self.down_ratio = down_ratio
        self.reshape = mindspore.ops.Reshape()
        self.ne = mindspore.ops.NotEqual()
        self.sum = mindspore.ops.ReduceSum(keep_dims=False)
        self.softmax = mindspore.ops.Softmax(axis=1)
        self.transpose = mindspore.ops.Transpose()
        self.mul = mindspore.ops.Mul()
        self.sort = mindspore.ops.Sort()
        self.le = mindspore.ops.LessEqual()
        self.min = mindspore.ops.Minimum()
        self.criterion = SoftmaxCrossEntropyLoss(num_cls=self.num_cls, ignore_label=self.ignore_label)
        self.cast = P.Cast()
        self.min = mindspore.ops.Minimum()
        self.max = mindspore.ops.Maximum()

    def construct(self, pred, target):
        """construct"""
        b, c, h, w = pred.shape
        target = self.reshape(target, (-1,))
        valid_mask = self.ne(target, self.ignore_label)
        target = target * valid_mask.astype("int64")
        num_valid = valid_mask.astype("int64").sum()

        prob = self.softmax(pred)
        prob = self.transpose(prob, (1, 0, 2, 3))
        prob = self.reshape(prob, (c, -1))

        if self.min_kept > num_valid:
            pass
        elif num_valid > 0:
            prob = masked_fill(prob, ~valid_mask, 1)
            ran = nn.Range(start=0, limit=len(target), delta=1)
            t = ran().astype("int64")
            mask_prob = prob[target.astype("int64"), t]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = self.sort(mask_prob)
                threshold_index = index[self.min(self.cast(len(index), mstype.int64), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = self.le(mask_prob, threshold)
                target = target * kept_mask.astype("int64")
                valid_mask = valid_mask * kept_mask.astype("int64")
                valid_mask = valid_mask.astype("bool")

        target = masked_fill(target, ~valid_mask, self.ignore_label)
        target = self.reshape(target, (b, h, w))

        return self.criterion(pred, target)


class CriterionOhemDSN(nn.Cell):
    """DSN : We need to consider two supervision for the models."""
    def __init__(self, args):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = 255
        self.ncls = args.num_classes
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_label=self.ignore_index, ncls=self.ncls,
                                                   thresh=args.ohem_thres, min_kept=args.ohem_keep)
        self.criterion2 = SoftmaxCrossEntropyLoss(num_cls=self.ncls, ignore_label=self.ignore_index)
        self.net = DualSeg_res101(num_classes=args.num_classes, is_train=True)
        self.h = args.input_size
        self.w = args.input_size

    def construct(self, image, target):
        """
        :param image: (1, 3, 832, 832)
        :param target: (1, 832, 832)
        :return:
        """
        preds = self.net(image)  # [Tensor1, Tensor2]
        target = target.astype("int64")
        resize1 = mindspore.ops.ResizeBilinear((self.h, self.w), align_corners=True)
        scale_pred = resize1(preds[0])  # (1, 19, 832, 832)
        loss1 = self.criterion1(scale_pred, target)
        scale_pred = resize1(preds[1])  # (1, 19, 832, 832)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2 * 0.4
