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
"""Triplet loss"""
import mindspore.ops as F
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
import mindspore.common.dtype as mstype


class PairwiseDistance(nn.Cell):
    # __constants__ = ['norm', 'eps', 'keepdim']
    # norm: float
    # eps: float
    # keepdim: bool
    def __init__(self, p=2, eps=1e-6, keepdim=False):
        super(PairwiseDistance, self).__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim
        self.square = P.Square()
        self.reduce_sum = P.ReduceSum()
        self.sqrt = P.Sqrt()

    def construct(self, x1, x2):
        output = x1 - x2 + self.eps
        square = self.square(output)
        all_sum = self.reduce_sum(square, 1)
        result = self.sqrt(all_sum)
        return result


class TripletLoss(nn.Cell):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance()
        self.op_mean = P.ReduceMean(keep_dims=True)
        self.clip_value_min = Tensor(0.0, mstype.float32)
        self.clip_value_max = Tensor(10000.0, mstype.float32)
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.squeeze = P.Squeeze(axis=-1)
        print("TripletLoss Created", flush=True)

    def construct(self, anchor, positive, negative, all_index):
        pos_dist = self.pdist(anchor, positive)
        neg_dist = self.pdist(anchor, negative)
        hinge_dist = F.clip_by_value(self.margin + pos_dist - neg_dist, self.clip_value_min, self.clip_value_max)
        all_index = self.squeeze(all_index)
        hinge_dist = hinge_dist * all_index
        index_num = self.reduce_sum(all_index, 0) + 1e-8
        hinge_dist_sum = self.reduce_sum(hinge_dist, 0)
        loss = hinge_dist_sum / index_num
        return loss
