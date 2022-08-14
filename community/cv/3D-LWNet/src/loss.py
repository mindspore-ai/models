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
"""define loss function for network"""
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.nn import LossBase
import mindspore.ops as ops
import numpy as np


class NllLoss(LossBase):
    """nllloss"""
    def __init__(self, reduction='mean', num_classes=1000):
        super(NllLoss, self).__init__(reduction)
        self.loss = ops.NLLLoss(reduction=reduction)
        self.weight = Tensor(np.ones(num_classes), mstype.float32)
        self.total_weight = num_classes

    def construct(self, logits, labels):
        out, weight = self.loss(logits, labels, self.weight)
        self.total_weight = weight
        return out
