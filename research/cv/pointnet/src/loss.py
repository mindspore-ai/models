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
"""Custom losses."""
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F
__all__ = ['PointnetLoss']


class PointnetLoss(nn.Cell):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, feature_transform, num_class=2):
        super(PointnetLoss, self).__init__()

        self.base_loss = NLLLoss()
        self.reshape = ops.Reshape()
        self.num_class = num_class
        self.trans_feat_loss = 0
        self.feature_transform = feature_transform

    def construct(self, *inputs):
        """construct"""
        preds, target = inputs
        preds = self.reshape(preds, (-1, self.num_class))
        target = self.reshape(target, (-1, 1))[:, 0] - 1
        target = target.astype('int32')
        loss = self.base_loss(preds, target)
        return loss

class NLLLoss(LossBase):
    '''
       NLLLoss function
    '''
    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__(reduction)
        self.one_hot = P.OneHot()
        self.reduce_sum = P.ReduceSum()

    def construct(self, logits, label):
        label_one_hot = self.one_hot(label, F.shape(logits)[-1], F.scalar_to_tensor(1.0), F.scalar_to_tensor(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return self.get_loss(loss)
