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
""" loss.py """

import mindspore.nn as nn
import mindspore.ops as P


class MarginRankingLoss(nn.Cell):
    """
    MarginRankingLoss
    """
    def __init__(self, margin=0, error_msg=None):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.error_msg = error_msg
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ge = P.GreaterEqual()
        self.sum = P.ReduceSum(keep_dims=True)
        self.mean = P.ReduceMean(keep_dims=True)

    def construct(self, input1, input2):
        """
        Args:
            input1: dist_an(anchor negative)
            input2: dist_ap(anchor positive)
        """
        # we want self.margin < dist_an - dist_ap
        # i.e. dist_ap - dist_an + self.margin < 0
        temp1 = self.sub(input2, input1)
        temp2 = self.add(temp1, self.margin)
        mask = self.ge(temp2, 0)

        loss = self.mean(temp2 * mask)
        return loss


class OriTripletLoss(nn.Cell):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, batch_size=64, error_msg=None):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.error_msg = error_msg
        self.ranking_loss = MarginRankingLoss(self.margin)

        self.pow = P.Pow()
        self.sum = P.ReduceSum(keep_dims=True)
        self.transpose = P.Transpose()
        self.mul = P.Mul()
        self.add = P.Add()
        self.sub = P.Sub()
        self.sqrt = P.Sqrt()
        self.equal = P.Equal()
        self.notequal = P.NotEqual()
        self.squeeze = P.Squeeze()
        self.unsqueeze = P.ExpandDims()
        self.max = P.ReduceMax(keep_dims=True)
        self.min = P.ReduceMin(keep_dims=True)
        self.matmul = P.MatMul()
        self.expand = P.BroadcastTo((batch_size, batch_size))

    def construct(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        # Compute pairwise distance, replace by the official when merged
        dist = self.pow(inputs, 2)
        dist = self.sum(dist, 1)
        dist = self.expand(dist)
        dist = self.add(dist, self.transpose(dist, (1, 0)))

        temp1 = self.matmul(inputs, self.transpose(inputs, (1, 0)))
        temp1 = self.mul(-2, temp1)
        dist = self.add(dist, temp1)
        # for numerical stability, clip_value_max=? why must set?
        dist = P.composite.clip_by_value(dist, clip_value_min=1e-12, clip_value_max=100000000)
        dist = self.sqrt(dist)

        # For each anchor, find the hardest positive and negative
        targets = self.expand(targets)
        mask_pos = self.equal(targets, self.transpose(targets, (1, 0)))
        mask_neg = self.notequal(targets, self.transpose(targets, (1, 0)))
        dist_ap = self.max(dist * mask_pos, 1).squeeze()
        dist_an = self.min(self.max(dist * mask_neg, 1) * mask_pos + dist, 1).squeeze()

        # Compute ranking hinge loss
        loss = self.ranking_loss(dist_an, dist_ap)

        return loss


class CenterTripletLoss(nn.Cell):
    """
    CenterTripletLoss
    """
    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.ori_tri_loss = OriTripletLoss(batch_size=batch_size // 4, margin=margin)
        self.unique = P.Unique()
        self.cat = P.Concat(0)
        self.mean = P.ReduceMean(False)
        self.chunk_ = P.Split(0, batch_size // 4)

    def construct(self, input_, label):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - label: ground truth labels with shape (num_classes)
        """

        dim = input_.shape[1]
        label_uni = self.unique(label)[0]
        targets = self.cat((label_uni, label_uni))
        label_num = len(label_uni)

        feat = self.chunk_(input_)
        center = []
        for i in range(label_num * 2):
            center.append(self.mean(feat[i], 0))
        input_ = self.cat(center).view((len(center), dim))
        loss = self.ori_tri_loss(input_, targets)

        return loss[0]
