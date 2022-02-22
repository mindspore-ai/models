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
"""Loss function."""
from mindspore import ops, nn


class JointsMSELoss(nn.Cell):
    """MSE loss for heatmaps.
    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=True, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.expand = ops.ExpandDims()

    def construct(self, output, target, target_weight):
        """Forward function."""
        if self.use_target_weight:
            target_weight = self.expand(target_weight, 2)
            target_weight = self.expand(target_weight, 3)
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)
        return loss
