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

import mindspore.numpy as np
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.ops.operations import Abs


class HuberLossWithWeight(nn.LossBase):
    """
    huber loss
    """

    def __init__(self):
        super(HuberLossWithWeight, self).__init__()
        self.abs = Abs()

    def construct(self, predictons, labels, weight=1.0, k=1.0):
        diff = predictons - labels
        abs_diff = self.abs(diff)
        k = np.array(k)
        losses = np.where(abs_diff < k, 0.5 * np.square(diff), k * abs_diff - 0.5 * k ** 2)
        return self.get_loss(losses, weight)


class MSELossWithWeight(nn.LossBase):
    """
    mse loss
    """

    def construct(self, base, target, weight=1.0):
        x = F.square(base - target)
        return self.get_loss(x, weight)


class WeightLoss(nn.LossBase):
    """
    weight loss
    """

    def construct(self, loss, weight=1.0):
        return self.get_loss(loss, weight)
