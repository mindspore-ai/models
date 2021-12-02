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
"""criterion function"""
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import numpy as mnp
from mindspore import ops as mops
from mindspore.ops import operations as P
from mindspore import Tensor



class SoftmaxCrossEntropyLoss(nn.Cell):
    """SoftmaxCrossEntropyLoss"""
    def __init__(self, num_cls, weight):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.unsqueeze = mops.ExpandDims()
        self.get_size = mops.Size()
        self.exp = mops.Exp()
        self.pow = mops.Pow()
        self.weight = weight

    def construct(self, pred, labels):
        """construct"""
        labels = self.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1,))
        pred = self.transpose(pred, (0, 2, 3, 1))
        pred = self.reshape(pred, (-1, self.num_cls))
        one_hot_labels = self.one_hot(labels, self.num_cls, self.on_value, self.off_value)
        pred = self.cast(pred, mstype.float32)
        num = self.get_size(labels)

        if self.weight is not None:
            weight = mnp.copy(self.weight)
            weight = self.cast(weight, mstype.float32)
            weight = self.unsqueeze(weight, 0)
            expand = mops.BroadcastTo(pred.shape)
            weight = expand(weight)
            weight_masked = weight[mnp.arange(num), labels]
            loss = self.ce(pred, one_hot_labels)
            loss = self.div(self.sum(loss * weight_masked), self.sum(weight_masked))
        else:
            loss = self.ce(pred, one_hot_labels)
            loss = self.div(self.sum(loss), num)
        return loss
