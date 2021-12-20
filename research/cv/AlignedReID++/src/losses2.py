"""relative loss functions and other functions"""
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

from __future__ import absolute_import
import numpy as np

import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as Co
from mindspore.common import dtype as mstype


class BatchEuclideanDist(nn.Cell):
    """compute the BatchEuclideanDist"""
    def __init__(self):
        super(BatchEuclideanDist, self).__init__()
        self.op_pow = ops.Pow()
        self.op_sum = ops.ReduceSum(keep_dims=True)
        self.transdim = (0, 2, 1)
        self.op_transpose = ops.Transpose()
        self.op_batmatmul = ops.BatchMatMul()
        self.min_value = Tensor(1e-12, mstype.float32)
        self.max_value = Tensor(1e+24, mstype.float32)
        self.op_sqrt = ops.Sqrt()
        self.op_broad1 = ops.BroadcastTo((32, 8, 8))
        self.op_broad2 = ops.BroadcastTo((32, 8, 8))

    def construct(self, x, y):
        """construct the BatchEuclideanDist"""
        if len(x.shape) != 3:
            print("error: len(x.shape) != 3")
        if len(y.shape) != 3:
            print("error: len(y.shape) != 3")
        if x.shape[0] != y.shape[0]:
            print("error: x.shape[0] != y.shape[0]")
        if x.shape[-1] != y.shape[-1]:
            print("error: x.shape[-1] != y.shape[-1]")

        out1 = self.op_pow(x, 2)
        out1 = self.op_sum(out1, -1)
        xx = self.op_broad1(out1)

        out2 = self.op_pow(y, 2)
        out2 = self.op_sum(out2, -1)
        yy = self.op_broad2(out2)
        yy = self.op_transpose(yy, self.transdim)

        dist = xx + yy

        y = self.op_transpose(y, self.transdim)
        output = self.op_batmatmul(x, y)
        dist = dist*1 + output*(-2)
        dist = Co.clip_by_value(dist, self.min_value, self.max_value)
        dist = self.op_sqrt(dist)
        return dist

class ShortestDist(nn.Cell):
    """compute the shortestdist"""
    def __init__(self):
        super(ShortestDist, self).__init__()
        self.op_minimum = ops.Minimum()

    def construct(self, dist_mat):
        """compute the shortestdist"""
        m = dist_mat.shape[0]
        n = dist_mat.shape[1]
        dist = []
        for i in range(m):
            dist.append([])
            for j in range(n):
                dist[i].append(0)

        for i in range(m):
            for j in range(n):
                if (i == 0) and (j == 0):
                    dist[i][j] = dist_mat[i, j]
                elif (i == 0) and (j > 0):
                    dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
                elif (i > 0) and (j == 0):
                    dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
                else:
                    dist[i][j] = self.op_minimum(dist[i - 1][j], dist[i][j - 1])+dist_mat[i, j]

        dist = dist[-1][-1]
        return dist

class EuclideanDist(nn.Cell):
    """compuet the euclideandist"""
    def __init__(self):
        super(EuclideanDist, self).__init__()
        self.op_pow = ops.Pow()
        self.op_sum = ops.ReduceSum(keep_dims=True)
        self.op_broad1 = ops.BroadcastTo((8, 8))
        self.op_broad2 = ops.BroadcastTo((8, 8))
        self.op_transpose = ops.Transpose()
        self.transdim = (1, 0)
        self.min_value = Tensor(1e-12, mstype.float32)
        self.max_value = Tensor(1e+24, mstype.float32)
        self.op_sqrt = ops.Sqrt()

    def construct(self, x, y):
        """compuet the euclideandist"""
        input_2 = 2.0
        msout3 = self.op_pow(x, input_2)
        msout3 = self.op_sum(msout3, 1)
        xx = self.op_broad1(msout3)

        msout4 = self.op_pow(y, input_2)
        msout4 = self.op_sum(msout4, 1)
        yy = self.op_broad2(msout4)
        yy = self.op_transpose(yy, self.transdim)

        dist = xx + yy
        dist = Co.clip_by_value(dist, self.min_value, self.max_value)
        dist = self.op_sqrt(dist)
        return dist

class BatchLocalDist(nn.Cell):
    """compuet the BatchLocalDist"""
    def __init__(self):
        super(BatchLocalDist, self).__init__()
        self.op_transpose = ops.Transpose()
        self.ShortestDist = ShortestDist()
        self.BatchEuclideanDist = BatchEuclideanDist()
        self.op_exp = ops.Exp()
        self.transdim = (1, 2, 0)

    def construct(self, x, y):
        """compuet the BatchLocalDist"""
        dist_mat = self.BatchEuclideanDist(x, y)
        msout = self.op_exp(dist_mat)
        dist_mat = (msout-1.)/(msout+1.)
        dist_mat = self.op_transpose(dist_mat, self.transdim)
        dist = self.ShortestDist(dist_mat)
        return dist

class HardExampleMining(nn.Cell):
    """hard example mining"""
    def __init__(self, return_inds=True, num_instances=4):
        super(HardExampleMining, self).__init__()
        self.broadcast = P.BroadcastTo((1, 32))
        self.transpose = P.Transpose()
        self.equal = P.Equal()
        self.notequal = P.NotEqual()
        self.select = P.Select()
        self.zeros = Tensor(np.zeros((32, 32)), mstype.float32)
        self.beforeind = Tensor(np.arange(0, 32), mstype.float32)
        self.broadcast_ind = P.BroadcastTo((32, 32))
        self.argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.argmin = P.ArgMinWithValue(axis=1, keep_dims=True)
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.idx_add = Tensor([0, 1, 2, 3], mstype.int32)
        self.broadcast_idx = P.BroadcastTo((32, 4))
        self.gatherd = P.GatherD()
        self.topk = P.TopK()
        self.sort = P.Sort()
        self.return_inds = return_inds
        self.num_instances = num_instances

    def construct(self, dist_mat, label):
        """hard example mining and bs is 32"""
        label_broad = self.broadcast(label)
        label_broad_transpose = self.transpose(label_broad, (1, 0))
        is_pos = self.equal(label_broad, label_broad_transpose)
        is_neg = self.notequal(label_broad, label_broad_transpose)
        ############get pos's dist_an and relative_d_idx############
        dist_mat_pos = self.select(is_pos, dist_mat, self.zeros)
        _, dist_pos_idx = self.topk(dist_mat_pos, self.num_instances)
        sorted_pos_idx_value, _ = self.sort(self.cast(dist_pos_idx, mstype.float16))
        sorted_pos_idx_value = self.cast(sorted_pos_idx_value, mstype.int32)
        dist_pos_mat_tmp = self.gatherd(dist_mat_pos, 1, sorted_pos_idx_value)
        relative_p_idx, dist_ap = self.argmax(dist_pos_mat_tmp)
        ############get neg's dist_an and relative_d_idx############
        dist_mat_neg = self.select(is_neg, dist_mat, self.zeros)
        _, dist_neg_idx = self.topk(dist_mat_neg, 32-self.num_instances)
        sorted_neg_idx_value, _ = self.sort(self.cast(dist_neg_idx, mstype.float16))
        sorted_neg_idx_value = self.cast(sorted_neg_idx_value, mstype.int32)
        dist_neg_mat_tmp = self.gatherd(dist_mat_neg, 1, sorted_neg_idx_value)
        relative_n_idx, dist_an = self.argmin(dist_neg_mat_tmp)
        if self.return_inds:
            ind = self.broadcast_ind(self.beforeind)
            ind_pos = self.select(is_pos, ind, self.zeros)
            _, ind_pos_idx = self.topk(ind_pos, self.num_instances)
            sorted_indpos_idx_value, _ = self.sort(self.cast(ind_pos_idx, mstype.float16))
            sorted_indpos_idx_value = self.cast(sorted_indpos_idx_value, mstype.int32)
            ind_pos_mat_tmp = self.gatherd(ind_pos, 1, sorted_indpos_idx_value)
            relative_p_idx = self.cast(relative_p_idx, mstype.int32)
            p_inds = self.gatherd(ind_pos_mat_tmp, 1, relative_p_idx)

            ind_neg = self.select(is_neg, ind, self.zeros)
            _, ind_neg_idx = self.topk(ind_neg, 32-self.num_instances)
            sorted_indneg_idx_value, _ = self.sort(self.cast(ind_neg_idx, mstype.float16))
            sorted_indneg_idx_value = self.cast(sorted_indneg_idx_value, mstype.int32)
            ind_neg_mat_tmp = self.gatherd(ind_neg, 1, sorted_indneg_idx_value)
            relative_n_idx = self.cast(relative_n_idx, mstype.int32)
            n_inds = self.gatherd(ind_neg_mat_tmp, 1, relative_n_idx)

            return dist_ap, dist_an, p_inds, n_inds
        return dist_ap, dist_an

class MarginRankingLoss(nn.Cell):
    """compute MarginRankingLoss"""
    def __init__(self, margin=0.3):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.op_maximum = ops.Maximum()
        self.op_mean = ops.ReduceMean()

    def construct(self, x1, x2, y):
        """compute MarginRankingLoss"""
        out = y*(-1)*(x1-x2)+self.margin
        out = self.op_maximum(out, 0)
        loss = self.op_mean(out)
        return loss

def DeepSupervision(criterion, xs, y):
    """DeepSupervision"""
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss

class CrossEntropyLoss(nn.Cell):
    """compute id loss"""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.crossentropy_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, inputs, targets):
        """compute id loss"""
        loss = self.crossentropy_loss(inputs, targets)
        return loss

class CrossEntropyLabelSmooth(nn.Cell):
    """compute id loss with labelsmoth"""
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.softmax = nn.Softmax(axis=1)
        self.log = mindspore.ops.Log()
        self.op_expanddims = ops.ExpandDims()
        self.op_ones = ops.Ones()
        self.op_scatter = ops.ScatterNd()
        self.op_mean = ops.ReduceMean()
        self.op_sum = ops.ReduceSum()
        self.cast = P.Cast()
        self.equal = P.Equal()
        self.broadcast_to = ops.BroadcastTo((32, self.num_classes))
        self.label_flag = Tensor(np.arange(0, self.num_classes, 1).astype(np.int32))

    def construct(self, inputs, targets):
        """compute id loss with labelsmoth"""
        log_probs = self.softmax(inputs)
        log_probs = self.log(log_probs)
        input_x = self.label_flag
        input_x = self.broadcast_to(input_x)
        input_y = self.op_expanddims(targets, 1)
        input_y = self.cast(input_y, mstype.int32)
        input_y = self.broadcast_to(input_y)

        targets = self.equal(input_x, input_y)
        targets = self.cast(targets, mstype.float32)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (- targets) * log_probs
        loss = self.op_mean(loss, 0)
        loss = self.op_sum(loss)

        return loss

class TripletLossAlignedReID(nn.Cell):
    """compute triplet loss with DMLI"""
    def __init__(self, margin=0.3, num_instances=4, mutual_flag=False):
        super(TripletLossAlignedReID, self).__init__()
        self.margin = margin
        self.ranking_loss = MarginRankingLoss(margin=margin)
        self.ranking_loss_local = MarginRankingLoss(margin=margin)
        self.mutual_flag = mutual_flag
        self.op_sum = ops.ReduceSum(keep_dims=True)
        self.op_pow = ops.Pow()
        self.op_sqrt = ops.Sqrt()
        self.op_transpose = ops.Transpose()
        self.op_matmul = ops.MatMul()
        self.shape = (32, 32)
        self.transdim = (1, 0)
        self.transdim2 = (0, 2, 1)
        self.op_broadcas = ops.BroadcastTo(self.shape)
        self.min_value = Tensor(1e-12, mindspore.float32)
        self.max_value = Tensor(1e+24, mindspore.float32)
        self.HardExampleMining = HardExampleMining(return_inds=True, num_instances=num_instances)
        self.BatchLocalDist = BatchLocalDist()
        self.op_onelike = ops.OnesLike()
        self.op_squeeze = ops.Squeeze()
        self.op_squeeze1 = ops.Squeeze(1)
        self.cast = P.Cast()

    def construct(self, inputs, targets, local_features):
        """compute triplet loss with DMLI"""
        dist = self.op_pow(inputs, 2)
        dist = self.op_sum(dist, 1)
        dist = self.op_broadcas(dist)
        transpose_dist = self.op_transpose(dist, self.transdim)
        dist = dist + transpose_dist
        transpose_inputs = self.op_transpose(inputs, self.transdim)
        temp = self.op_matmul(inputs, transpose_inputs)
        temp = temp*(-2)
        dist = temp+dist
        dist = ops.clip_by_value(dist, clip_value_min=self.min_value, clip_value_max=self.max_value)
        dist = self.op_sqrt(dist)

        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an, p_inds, n_inds = self.HardExampleMining(dist, targets)
        local_features = self.op_transpose(local_features, self.transdim2)
        p_inds = self.cast(p_inds, mindspore.int32)
        n_inds = self.cast(n_inds, mindspore.int32)
        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]

        p_local_features = self.op_squeeze1(p_local_features)
        n_local_features = self.op_squeeze1(n_local_features)

        local_dist_ap = self.BatchLocalDist(local_features, p_local_features)
        local_dist_an = self.BatchLocalDist(local_features, n_local_features)

        # Compute ranking hinge loss
        dist_an = self.op_squeeze(dist_an)
        dist_ap = self.op_squeeze(dist_ap)
        y = self.op_onelike(dist_an)
        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        local_loss = self.ranking_loss_local(local_dist_an, local_dist_ap, y)
        if self.mutual_flag:
            return global_loss+local_loss, dist
        return global_loss, local_loss

class AllLoss(nn.Cell):
    """the loss function interface"""
    def __init__(self, num_classes=10, margin=0.3, labelsmoth=False, num_instances=4):
        super(AllLoss, self).__init__()
        self.cla1 = CrossEntropyLoss()
        self.cla2 = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.cla4 = TripletLossAlignedReID(margin=margin, num_instances=num_instances)
        self.op_squeeze = ops.Squeeze(3)
        self.labelsmoth = labelsmoth

    def construct(self, outputs, pids, features, local_features):
        """compute loss"""
        if isinstance(outputs, tuple):
            xent_loss = 0
            for x in outputs:
                xent_loss = xent_loss+self.cla2(x, pids)
        else:
            if self.labelsmoth:
                xent_loss = self.cla2(outputs, pids)
            else:
                xent_loss = self.cla1(outputs, pids)

        local_features = self.op_squeeze(local_features)
        if isinstance(features, tuple):
            global_loss = 0.
            local_loss = 0.
            for x in features:
                lossg, lossl = self.cla4(features, pids, local_features)
                global_loss += lossg
                local_loss += lossl
        else:
            global_loss, local_loss = self.cla4(features, pids, local_features)

        loss = xent_loss + global_loss + local_loss
        return loss

class CustomWithLossCell(nn.Cell):
    """self-define withlosscell function"""
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, imgs, pids, camids):
        """self-define withlosscell function"""
        outputs, features, local_features = self._backbone(imgs)
        return self._loss_fn(outputs, pids, features, local_features)
