# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Define loss"""
import mindspore.nn as nn


class SoftmaxCrossEntropyLoss(nn.Cell):
    """
    Define the loss use auxiliary
    """

    def __init__(self, auxiliary, auxiliary_weight):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction='mean')

    def construct(self, data, labels):
        if self.auxiliary and self.training:
            logits, logits_aux = data
            loss = self.criterion(logits, labels)
            loss_aux = self.criterion(logits_aux, labels)
            loss += self.auxiliary_weight * loss_aux
        else:
            logits = data
            loss = self.criterion(logits, labels)
        return loss
