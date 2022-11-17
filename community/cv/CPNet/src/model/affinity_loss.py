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
""" AFFINITY LOSS """
from mindspore import Tensor
import mindspore.ops as ops
from mindspore import nn
import mindspore.numpy as np
from mindspore import dtype as mstype


class AffinityLoss(nn.Cell):
    """ affinity loss """
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = 21
        self.expanddims = ops.ExpandDims()
        self.bmm = ops.BatchMatMul()
        self.bce = ops.BinaryCrossEntropy()
        self.bceloss = nn.BCELoss(reduction='mean')
        self.ones = ops.OnesLike()
        self.rnn = ops.ResizeNearestNeighbor((60, 60))
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = ops.Cast()

    def ConstructIdealaffinityMap(self, label, label_size):
        """ construct ideal affinity map """
        label = self.expanddims(label, 1)
        scaled_labels = self.rnn(label)
        scaled_labels = scaled_labels.squeeze(axis=1).astype('int64')
        scaled_labels[scaled_labels == 255] = self.num_classes
        one_hot_labels = self.onehot(scaled_labels, self.num_classes + 1, self.on_value, self.off_value)
        one_hot_labels = one_hot_labels.reshape((one_hot_labels.shape[0], -1, self.num_classes + 1))
        ideal_affinity_map = self.bmm(one_hot_labels, one_hot_labels.transpose((0, 2, 1)))
        return ideal_affinity_map

    def construct(self, cls_score, label):
        """ calculate affinity loss """
        IdealaffinityMap = self.ConstructIdealaffinityMap(label, [60, 60])
        bce_loss = self.bceloss(cls_score, IdealaffinityMap)
        diagonal_matrix = (1 - np.eye(IdealaffinityMap.shape[1]))
        target = np.multiply(diagonal_matrix, IdealaffinityMap)
        recall_numerator = np.sum(cls_score * target.squeeze(), axis=2)
        denominator = np.sum(IdealaffinityMap, axis=2)
        denominator = np.where(denominator <= 0, self.ones(denominator), denominator)
        recall_part = np.divide(recall_numerator, denominator)
        recall_label = self.ones(recall_part)
        recall_loss = self.bceloss(recall_part, recall_label)
        spec_numerator = np.sum(np.multiply(1 - cls_score, 1 - IdealaffinityMap), axis=2)
        denominator = np.sum(1 - IdealaffinityMap, axis=2)
        denominator = np.where(denominator <= 0, self.ones(denominator), denominator)
        spec_part = np.divide(spec_numerator, denominator)
        spec_label = self.ones(spec_part)
        spec_loss = self.bceloss(spec_part, spec_label)
        precision_numerator = np.sum(np.multiply(cls_score, IdealaffinityMap), axis=2)
        denominator = np.sum(cls_score, axis=2)
        denominator = np.where(denominator <= 0, self.ones(denominator), denominator)
        precision_part = np.divide(precision_numerator, denominator)
        precision_label = self.ones(precision_part)
        precision_loss = self.bceloss(precision_part, precision_label)
        global_loss = (recall_loss + spec_loss + precision_loss) / 60
        affinity_loss = bce_loss + global_loss
        return affinity_loss
