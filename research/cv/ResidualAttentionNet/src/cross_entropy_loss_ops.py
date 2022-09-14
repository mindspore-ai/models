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
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.ops import operations as P
from mindspore.nn import Cell


class CrossEntropyLoss(Cell):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()

        self.reduce_mean = P.ReduceMean()
        self.cross_entropy = SoftmaxCrossEntropyWithLogits()
        self.reduction = reduction

    def construct(self, logits, label):
        loss = self.cross_entropy(logits, label)
        if self.reduction == 'mean':
            loss = self.reduce_mean(loss, (-1,))
        return loss
