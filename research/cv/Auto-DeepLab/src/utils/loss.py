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
# ===========================================================================
"""Loss function"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import stop_gradient


class CrossEntropyLoss(nn.Cell):
    """CrossEntropyLoss"""
    def __init__(self,
                 ignore_label=255):
        super(CrossEntropyLoss, self).__init__()

        self.cast = ops.Cast()
        self.scast = ops.ScalarCast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

        self.not_equal = ops.NotEqual()
        self.equal = ops.Equal()

        self.mul = ops.Mul()
        self.sum = ops.ReduceSum(False)
        self.div = ops.RealDiv()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

        self.ignore_label = ignore_label

    def construct(self, logits, labels):
        """construct"""
        num_cls = logits.shape[1]
        labels_int32 = self.cast(labels, mindspore.int32)
        labels_int = self.reshape(labels_int32, (-1,))
        logits_1 = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_1, (-1, num_cls))

        weights_1 = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights_1, mindspore.float32)

        _ce_loss = self.ce(logits_, labels_int)
        weighted_ce_loss = self.mul(weights, _ce_loss)
        ce_loss = self.div(self.sum(weighted_ce_loss), self.sum(weights))
        return ce_loss


class OhemCELoss(nn.Cell):
    """OhemCELoss"""
    def __init__(self,
                 thresh,
                 n_min,
                 ignore_label=255):
        super(OhemCELoss, self).__init__()

        self.cast = ops.Cast()
        self.scast = ops.ScalarCast()

        self._thresh = self.scast(thresh, mindspore.float32)
        self._n_min = n_min
        self._ignore_label = ignore_label

        self.topk = ops.TopK(sorted=True)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.not_equal = ops.NotEqual()
        self.equal = ops.Equal()
        self.min = ops.Minimum()
        self.mul = ops.Mul()
        self.sum = ops.ReduceSum(False)
        self.div = ops.RealDiv()
        self.gather = ops.GatherNd()

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, logits, labels):
        """construct"""
        _, c, _, _ = logits.shape
        num_classes = c

        labels_0 = self.cast(labels, mindspore.int32)
        labels_1 = self.reshape(labels_0, (-1,))
        logits_0 = self.transpose(logits, (0, 2, 3, 1))
        logits_1 = self.reshape(logits_0, (-1, num_classes))

        keep_mask_0 = self.not_equal(labels_1, self._ignore_label)
        keep_mask_1 = self.cast(keep_mask_0, mindspore.float32)

        pix_losses = self.ce(logits_1, labels_1)
        masked_pixel_losses = self.mul(keep_mask_1, pix_losses)

        top_k_losses, _ = self.topk(masked_pixel_losses, self._n_min)
        thresh = self.min(self._thresh, top_k_losses[self._n_min - 1:self._n_min:1])

        ohem_mask = self.cast(masked_pixel_losses >= thresh, mindspore.float32)
        ohem_mask = stop_gradient(ohem_mask)
        ohem_loss = self.mul(ohem_mask, masked_pixel_losses)
        total_loss = self.sum(ohem_loss)
        num_present = self.sum(ohem_mask)
        loss = self.div(total_loss, num_present)

        return loss


def build_criterion(args):
    """build_criterion"""
    print("=> Trying build {:}loss".format(args.criterion))
    if args.criterion == 'ce':
        loss = CrossEntropyLoss(ignore_label=args.ignore_label)
    elif args.criterion == 'ohemce':
        loss = OhemCELoss(args.thresh, args.n_min, args.ignore_label)
    else:
        raise ValueError('unknown criterion : {:}'.format(args.criterion))
    return loss
