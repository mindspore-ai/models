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
""" Triplet Loss """

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class MarginRankingLoss(nn.Cell):
    """ Creates a criterion that measures the loss given
    inputs x1, x2, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor y (containing 1 or -1).

    If y = 1 then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for y = -1.

    The loss function for each pair of samples in the mini-batch is:

        loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin

    Args:
        margin: margin value
        reduction: reduction function
    """
    def __init__(self, margin=0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.maximum = ops.Maximum()
        if reduction == 'mean':
            self.reduction = ops.ReduceMean(keep_dims=False)
        elif reduction == 'sum':
            self.reduction = ops.ReduceSum(keep_dims=False)
        else:
            raise ValueError(f'Unknown reduction {reduction}')

    def construct(self, input1, input2, target):
        """ Forward """
        diff = self.maximum(0, -target * (input1 - input2) + self.margin)
        return self.reduction(diff)


class TripletLoss(nn.Cell):
    """ Triplet loss with MarginRankingLoss or SoftMarginLoss

    Args:
        margin: margin value (if 0 then use SoftMarginLoss)
        reduction: reduction function
    """
    def __init__(self, margin=0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.positive = mindspore.Parameter(1, requires_grad=False)
        if margin > 0:
            self.ranking_loss = MarginRankingLoss(margin=margin, reduction=reduction)
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction=reduction)

    def construct(self, dist_ap, dist_an):
        """ Forward """
        if self.margin > 0:
            loss = self.ranking_loss(dist_an, dist_ap, self.positive)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, self.positive)

        return loss
