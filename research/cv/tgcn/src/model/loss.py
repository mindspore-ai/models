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
"""
T-GCN loss cell
"""
import mindspore.nn as nn
import mindspore.numpy as np


class TGCNLoss(nn.Cell):
    """
    Custom T-GCN loss cell
    """

    def construct(self, predictions, targets):
        """
        Calculate loss

        Args:
            predictions(Tensor): predictions from models
            targets(Tensor): ground truth

        Returns:
            loss: loss value
        """
        targets = targets.reshape((-1, targets.shape[2]))
        return np.sum((predictions - targets) ** 2) / 2
