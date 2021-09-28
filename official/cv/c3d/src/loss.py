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

from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P
import mindspore.nn as nn

class Max_Entropy(LossBase):

    def __init__(self, reduction='mean'):
        super(Max_Entropy, self).__init__(reduction)
        self.logsoftmax = nn.LogSoftmax(axis=1)
        self.sum = P.ReduceSum()
        self.mean = P.ReduceMean()

    def construct(self, predictions, targets):
        log = self.logsoftmax(predictions)
        temp_ = self.sum(-targets * log, 1)
        loss = self.mean(temp_)

        return self.get_loss(loss)
