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
"""
python loss.py
"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor


class LabelTransform(nn.Cell):
    """ label transform """

    def __init__(self, num_classes=5):
        super(LabelTransform, self).__init__()
        self.num_classes = num_classes
        self.T = Tensor(1, mindspore.int32)
        self.F = Tensor(0, mindspore.int32)

    def construct(self, label):
        """ Construct  label transform """
        label1 = mnp.where(label > 0, self.T, self.F)
        flair_t2_gt_node = ops.one_hot(label1, 2, 1.0, 0.0)
        t1_t1ce_gt_node = ops.one_hot(label, self.num_classes, 1.0, 0.0)
        return flair_t2_gt_node, t1_t1ce_gt_node


class SegmentationLoss(nn.Cell):
    """ segmentation loss """

    def __init__(self, num_classes=5):
        super(SegmentationLoss, self).__init__()
        self.label_transform = LabelTransform(num_classes)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
        self.num_classes = num_classes

    def construct(self, flair_t2_score, t1_t1ce_score, label):
        """ Construct segmentation loss """
        flair_t2_gt_node, t1_t1ce_gt_node = self.label_transform(label)
        flair_t2_score = self.transpose(flair_t2_score, (0, 2, 3, 4, 1))
        flair_t2_gt_node = self.reshape(flair_t2_gt_node, (-1, 2))
        flair_t2_score = self.reshape(flair_t2_score, (-1, 2))
        t1_t1ce_score = self.transpose(t1_t1ce_score, (0, 2, 3, 4, 1))
        t1_t1ce_gt_node = self.reshape(t1_t1ce_gt_node, (-1, self.num_classes))
        t1_t1ce_score = self.reshape(t1_t1ce_score, (-1, self.num_classes))
        flair_t2_loss = self.loss(labels=flair_t2_gt_node, logits=flair_t2_score)
        t1_t1ce_loss = self.loss(labels=t1_t1ce_gt_node, logits=t1_t1ce_score)
        loss = flair_t2_loss + t1_t1ce_loss
        return loss


class NetWithLoss(nn.Cell):
    """ NetWithLoss """

    def __init__(self, network, num_classes=5):
        super(NetWithLoss, self).__init__()
        self.net = network
        self.loss_func = SegmentationLoss(num_classes)
        self.num_classes = num_classes

    def construct(self, flair_t2_node, t1_t1ce_node, label):
        """ Construct NetWithLoss """
        flair_t2_score, t1_t1ce_score = self.net(flair_t2_node, t1_t1ce_node)
        loss = self.loss_func(flair_t2_score, t1_t1ce_score, label)
        return loss
