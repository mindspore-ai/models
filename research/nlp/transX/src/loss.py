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
"""
Loss function for a model.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class TripletsMarginLoss(nn.Cell):
    """Calculate the margin loss between original and corrupted triplets

    Args:
        negative_rate (int): how many negative triplets are used for a single original triplet
        margin (float): Margin
    """

    def __init__(self, negative_rate, margin):
        super().__init__()

        if negative_rate < 1:
            raise ValueError(
                f'Triplets margin loss needs corrupted triplets. '
                f'Set the negative sampling rate >= 1'
            )

        self.negative_rate = negative_rate
        self.margin = ms.Parameter(ms.Tensor(margin, ms.float32), requires_grad=False)

    def construct(self, scores):
        """Construct a forward graph"""

        # We expect that the scores have the shape [batch_size * (1 + negative_rate)]
        scores_reshaped = scores.view(-1, 1 + self.negative_rate)

        p_score = scores_reshaped[:, :1]
        n_score = scores_reshaped[:, 1:]

        loss = ops.maximum(p_score - n_score, -self.margin).mean() + self.margin

        return loss


class LossWrapperCell(nn.Cell):
    """Loss Wrapper Cell using with the mindspore.Model

    Args:
        network (nn.Cell): Network for predicting scores
        loss_fn (nn.Cell): Loss function
    """
    def __init__(self, network, loss_fn):
        super().__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn

    def construct(self, batched_triplets):
        """Construct a forward graph"""
        scores = self._network(batched_triplets)
        return self._loss_fn(scores)
