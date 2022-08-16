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
"""Triplet loss with hard positive/negative mining"""
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor


class MarginRankingLoss(nn.Cell):
    """function MarginRankingLoss"""
    def __init__(self, margin=0.0, reduction='mean'):
        super(MarginRankingLoss, self).__init__()
        self.reduction = reduction
        self.margin = margin
        self.sum = ops.ReduceSum(keep_dims=False)

    def construct(self, input1, input2, target):
        output = ops.Maximum()(0, -target*(input1 - input2) + self.margin)
        if self.reduction == 'mean':
            output = np.mean(output)
        elif self.reduction == 'sum':
            output = self.sum(output, 0)

        return output

class addmm(nn.Cell):
    """function _addmm"""
    def construct(self, mat, alpha, beta, mat1, mat2):
        out = ops.matmul(mat1, mat2)

        return mat * alpha + out * beta

class TripletLoss(nn.Cell):
    """Triplet loss with hard positive/negative mining"""
    def __init__(self, batch_size, margin=0.3):
        super(TripletLoss, self).__init__()
        self.addmm = addmm()
        self.pow = ops.Pow()
        self.equal = ops.Equal()
        self.cast = ops.Cast()
        self.select = ops.Select()
        self.reducemax = ops.ReduceMax()
        self.reducemin = ops.ReduceMin()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.ranking_loss = MarginRankingLoss(margin=margin)
        self.expand = ops.BroadcastTo((batch_size, batch_size))
        self.zeros = Tensor(np.zeros((batch_size, batch_size)).astype(np.float32))
        self.maxs = Tensor(np.full((batch_size, batch_size), 65535).astype(np.float32))

    def construct(self, inputs, targets):
        """TripletLoss construct"""
        inputs_ = self.pow(inputs, 2)
        inputs_ = self.sum(inputs_, 1)

        dist = self.expand(inputs_)  # (32, 32)
        dist = dist + dist.T
        dist = self.addmm(dist, 1, -2, inputs, inputs.T)
        dist = ops.clip_by_value(dist, clip_value_min=1e-12, clip_value_max=np.amax(dist))
        dist = ops.Sqrt()(dist)
        targets = self.cast(targets, mstype.float32)
        mask = self.equal(self.expand(targets), self.expand(targets).T)
        dist_ap = self.select(mask, dist, self.zeros)
        mask_zeros = self.equal(self.cast(mask, mstype.int32), self.zeros)
        dist_an = self.select(mask_zeros, dist, self.maxs)
        dist_ap = self.reducemax(dist_ap, 1)
        dist_an = self.reducemin(dist_an, 1)
        y = np.ones_like((dist_an))

        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
