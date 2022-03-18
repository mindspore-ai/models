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
"""Loss functions."""
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as F
from mindspore.common.tensor import Tensor
from mindspore import dtype as mstype
from mindspore.nn.loss.loss import LossBase


class CrossEntropyWithLogits(LossBase):
    """
    Cross-entropy loss function for semantic segmentation,
    and different classes have the same weight.
    """
    def __init__(self, num_classes=19, ignore_label=255, image_size=None):
        super(CrossEntropyWithLogits, self).__init__()
        self.resize = F.ResizeBilinear(image_size)
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.argmax = P.Argmax(output_type=mstype.int32)
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """Loss construction."""
        logits = self.resize(logits)
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_classes))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_classes, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))

        return loss


class CrossEntropyWithWeights(LossBase):
    """
    Cross-entropy loss function for semantic segmentation,
    and different classes have different weights.
    """
    def __init__(self, num_classes=19, ignore_label=255, image_size=None, weights=None):
        super(CrossEntropyWithWeights, self).__init__()
        self.resize = F.ResizeBilinear(image_size)
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.zeros = F.Zeros()
        self.fill = F.Fill()
        self.equal = F.Equal()
        self.select = F.Select()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.argmax = P.Argmax(output_type=mstype.int32)
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        if weights:
            self.weights = weights
        else:
            self.weights = [1.0] * self.num_classes

    def construct(self, logits, labels):
        """Loss construction."""
        logits = self.resize(logits)
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_classes))
        labels_float = self.cast(labels_int, mstype.float32)
        weights = self.zeros(labels_float.shape, mstype.float32)
        for i in range(self.num_classes):
            fill_weight = self.fill(mstype.float32, labels_float.shape, self.weights[i])
            equal_ = self.equal(labels_float, i)
            weights = self.select(equal_, fill_weight, weights)
        one_hot_labels = self.one_hot(labels_int, self.num_classes, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))

        return loss
