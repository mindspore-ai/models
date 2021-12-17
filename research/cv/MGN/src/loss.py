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
""" Aligned ReID losses """

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from src.triplet_loss import TripletLoss


def euclidean_dist(x, y, min_val):
    """ Euclidean distance between each pair of vectors from x and y

    x and y are matrices:
    x = (x_1, | x_2 | ... | x_m).T
    y = (y_1, | y_2 | ... | y_n).T

    Where x_i and y_j are vectors. We calculate the distances between each pair x_i and y_j.

    res[i, j] = dict(x_i, y_j)

    res will have the shape [m, n]

    For calculation we use the formula x^2 - 2xy + y^2.

    Clipped to prevent zero distances for numerical stability.
    """
    m, n = x.shape[0], y.shape[0]
    xx = ops.pows(x, 2).sum(axis=1, keepdims=True).repeat(n, axis=1)
    yy = ops.pows(y, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
    dist = xx + yy

    dist = 1 * dist - 2 * ops.dot(x, y.transpose())

    # Avoiding zeros for numerical stability
    dist = ops.maximum(
        dist,
        min_val,
    )
    dist = ops.sqrt(dist)
    return dist


def normalize(x, axis=-1):
    norm = ops.sqrt(ops.pows(x, 2).sum(axis=axis, keepdims=True))
    x_normalized = 1. * x / (norm + 1e-12)
    return x_normalized


def hard_example_mining(dist_mat, labels):
    """ Search min negative and max positive distances

    Args:
        dist_mat: distance matrix
        labels: real labels

    Returns:
        distance to positive indices
        distance to negative indices
        positive max distance indices
        negative min distance indices

    """
    def get_max(dist_mat_, idxs, inv=False):
        """ Search max values in distance matrix (min if inv=True) """
        dist_mat_cp = dist_mat_.copy()
        if inv:
            dist_mat_cp = -dist_mat_cp
        # fill distances for non-idxs values as min value
        dist_mat_cp[~idxs] = dist_mat_cp.min() - 1
        pos_max = dist_mat_cp.argmax(axis=-1)
        maxes = dist_mat_.take(pos_max, axis=-1).diagonal()
        return pos_max, maxes

    n = dist_mat.shape[0]

    labels_sq = ops.expand_dims(labels, -1).repeat(n, axis=-1)

    # shape [n, n]
    is_pos = ops.equal(labels_sq, labels_sq.T)  # Same id pairs
    is_neg = ops.not_equal(labels_sq, labels_sq.T)  # Different id pairs

    p_inds, dist_ap = get_max(dist_mat, is_pos)  # max distance for positive and corresponding ids
    n_inds, dist_an = get_max(dist_mat, is_neg, inv=True)  # min distance for negative and corresponding ids

    return dist_ap, dist_an, p_inds, n_inds


def global_loss(tri_loss, global_feat, labels, min_val, normalize_feature=False):
    """ Global loss

    Args:
        tri_loss: triplet loss
        global_feat: global features
        labels: real labels
        normalize_feature: flag to normalize features
        min_val: value to cut distance from below
    Returns:
        loss value
        positive max distance indices
        negative min distance indices
        distance to positive indices
        distance to negative indices
        distance matrix
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat, min_val)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat


class MGNLoss(nn.LossBase):
    """ Combined global, local and Cross Entropy loss

    Args:
        margin: margin for global MarginRankingLoss
        g_loss_weight: weight of global loss
        id_loss_weight: weight of identity loss
        normalize_feature: flag to normalize features
        reduction: reduction function
    """
    def __init__(
            self,
            margin,
            g_loss_weight,
            id_loss_weight,
            normalize_feature=False,
            reduction='mean',
    ):
        super().__init__(reduction)
        self.g_loss_weight = g_loss_weight
        self.id_loss_weight = id_loss_weight

        self.g_tri_loss, self.l_tri_loss, self.id_criterion = None, None, None

        self.g_tri_losses = nn.CellList([
            TripletLoss(margin=margin) for _ in range(3)
        ])

        if self.id_loss_weight > 0:
            self.id_criterions = nn.CellList([
                nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean') for _ in range(8)
            ])

        self.normalize_feature = normalize_feature
        self.mean = ops.ReduceMean()
        self.stack = ops.Stack()

        self.min_val = mindspore.Tensor(1e-12, mindspore.float32)

    def construct(self, logits, labels):
        """ Forward """
        losses = []
        for ii in range(3):
            loss, _, _, _, _, _ = global_loss(
                self.g_tri_losses[ii],
                logits[ii+1],
                labels,
                normalize_feature=self.normalize_feature,
                min_val=self.min_val,
            )
            losses.append(loss)

        loss = self.mean(self.stack(losses)) * self.g_loss_weight

        if self.id_loss_weight > 0:
            losses = []
            for ii in range(8):
                id_loss = self.id_criterions[ii](logits[ii+4], labels)
                losses.append(id_loss)

            loss += self.mean(self.stack(losses)) * self.id_loss_weight
        return loss
