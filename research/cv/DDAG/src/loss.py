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

"""loss.py"""
import mindspore as ms
import mindspore.ops as P

from mindspore import nn


class MarginRankingLoss(nn.Cell):
    """
    class of MarginRankingLoss
    """
    def __init__(self, margin=0):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ge = P.GreaterEqual()
        self.sum = P.ReduceSum(keep_dims=True)
        self.mean = P.ReduceMean(keep_dims=True)

    def construct(self, input1, input2, y):
        temp1 = - self.sub(input1, input2)
        temp2 = self.mul(temp1, y)
        temp3 = self.add(temp2, self.margin)
        temp3_mask = self.ge(temp3, 0)

        loss = self.mean(temp3 * temp3_mask)
        return loss


class OriTripletLoss(nn.Cell):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, batch_size=64):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
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
        self.cat = P.Concat()
        self.ones_like = P.OnesLike()
        self.squeeze = P.Squeeze()
        self.unsqueeze = P.ExpandDims()
        self.max = P.ReduceMax(keep_dims=True)
        self.min = P.ReduceMin(keep_dims=True)
        self.cat = P.Concat()
        self.matmul = P.MatMul()
        self.expand = P.BroadcastTo((batch_size, batch_size))
        self.cast = P.Cast()

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
        dist = P.composite.clip_by_value(
            dist, clip_value_min=1e-12, clip_value_max=100000000)
        dist = self.sqrt(dist)

        # For each anchor, find the hardest positive and negative
        targets = self.expand(targets)
        mask_pos = self.cast(self.equal(
            targets, self.transpose(targets, (1, 0))), ms.int8)
        mask_neg = self.cast(self.notequal(
            targets, self.transpose(targets, (1, 0))), ms.int8)
        dist_ap = self.max(dist * mask_pos, 1).squeeze()
        dist_an = self.min(self.max(dist * mask_neg, 1)
                           * mask_pos + dist, 1).squeeze()

        # Compute ranking hinge loss
        y = self.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss[0]


class CenterTripletLoss(nn.Cell):
    """
    class of center triplet loss
    """

    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.ori_tri = OriTripletLoss(
            batch_size=batch_size // 4, margin=margin)
        self.unique = P.Unique()
        self.cat = P.Concat()
        self.mean = P.ReduceMean(keep_dims=False)

    def construct(self, input_, label):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - label: ground truth labels with shape (num_classes)
        """

        # dim = input_.shape[1]
        #################################
        # The following 3 lines can work normally in PYNATIVE_MODE,
        # but have problems in GRAPH_MODE, due to different behavier
        # of Unique() operation under two modes. We have reported this
        # to official and wait for future fixes.
        label_uni = self.unique(label)[0]
        targets = self.cat([label_uni, label_uni])  # 2class_num
        label_num = label_uni.shape[0]
        #################################

        input_trans = input_.transpose()  # [2048 , 64]
        # [2048 , 16, 4]
        input_trans = input_trans.view(
            (input_trans.shape[0], 2 * label_num, input_trans.shape[1] // (2 * label_num)))
        centers = P.ReduceMean()(input_trans, 2)
        new_input = centers.transpose()
        loss = self.ori_tri(new_input, targets)

        return loss
