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
        dist_mat_ = dist_mat_.copy()
        if inv:
            dist_mat_ = -dist_mat_
        # fill distances for non-idxs values as min value
        dist_mat_[~idxs] = dist_mat_.min() - 1
        pos_max = dist_mat_.argmax(axis=-1)
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


def batch_euclidean_dist(x, y, min_val):
    """ euclidean_dist function over batch

    x and y are batches of matrices x' and y':
    x' = (x'_1, | x'_2 | ... | x'_m).T
    y' = (y'_1, | y'_2 | ... | y'_n).T

    Where x_i and y_j are vectors. We calculate the distances between each pair x_i and y_j.

    res'[i, j] = dict(x'_i, y'_j)

    res (batch of res') will have the shape [batch_size, m, n]

    For calculation we use the formula x^2 - 2xy + y^2.

    Clipped to prevent zero distances for numerical stability.
    """
    _, m, _ = x.shape
    _, n, _ = y.shape

    # shape [N, m, n]
    xx = ops.pows(x, 2).sum(-1, keepdims=True).repeat(n, axis=-1)
    yy = ops.pows(y, 2).sum(-1, keepdims=True).repeat(m, axis=-1).transpose(0, 2, 1)
    dist = xx + yy

    dist = 1 * dist - 2 * ops.batch_dot(x, y.transpose(0, 2, 1))

    # Avoiding zeros for numerical stability
    dist = ops.maximum(
        dist,
        min_val,
    )
    dist = ops.sqrt(dist)
    return dist


def shortest_dist(dist_mat):
    """Parallel version.
    Args:
      dist_mat: tensor, available shape:
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
      dist: three cases corresponding to `dist_mat`:
        1) scalar
        2) tensor, with shape [N]
        3) tensor, with shape [*]
    """
    m, n = dist_mat.shape[:2]
    # Just offering some reference for accessing intermediate distance.
    # dist = [[0] * n for _ in range(m)]
    i = 0
    dist = []
    while i < m:
        i += 1
        dist.append([0 for _ in range(n)])

    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = ops.minimum(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]
    return dist


def batch_local_dist(x, y, min_val):
    """ Compute local distance for batch
    Args:
      x: pytorch Variable, with shape [N, m, d]
      y: pytorch Variable, with shape [N, n, d]
      min_val: Minimal distance (for preventing division by zero)
    Returns:
      dist: pytorch Variable, with shape [N]
    """
    # shape [N, m, n]
    dist_mat = batch_euclidean_dist(x, y, min_val)
    dist_mat = (ops.exp(dist_mat) - 1.) / (ops.exp(dist_mat) + 1.)
    # shape [N]
    dist = shortest_dist(dist_mat.transpose(1, 2, 0))
    return dist


def local_loss(
        tri_loss,
        local_feat,
        p_inds=None,
        n_inds=None,
        normalize_feature=False,
        min_val=None,
):
    """ Local loss

    Args:
        tri_loss: triplet loss
        local_feat: local features
        p_inds: positive max distance indices
        n_inds: negative min distance indices
        normalize_feature: flag to normalize features
        min_val: value to cut distance from below

    Returns:
        loss value
        distance to positive indices
        distance to negative indices
    """
    if normalize_feature:
        local_feat = normalize(local_feat, axis=-1)
    dist_ap = batch_local_dist(local_feat, local_feat[p_inds], min_val)
    dist_an = batch_local_dist(local_feat, local_feat[n_inds], min_val)
    loss = tri_loss(dist_ap, dist_an)
    return loss, dist_ap, dist_an


class ReIDLoss(nn.LossBase):
    """ Combined global, local and Cross Entropy loss

    Args:
        global_margin: margin for global MarginRankingLoss
        local_margin: margin for local MarginRankingLoss
        g_loss_weight: weight of global loss
        l_loss_weight: weight of local loss
        id_loss_weight: weight of identity loss
        normalize_feature: flag to normalize features
        reduction: reduction function
    """
    def __init__(
            self,
            global_margin,
            local_margin,
            g_loss_weight,
            l_loss_weight,
            id_loss_weight,
            normalize_feature=False,
            reduction='mean',
    ):
        super().__init__(reduction)
        self.g_loss_weight = g_loss_weight
        self.l_loss_weight = l_loss_weight
        self.id_loss_weight = id_loss_weight

        self.g_tri_loss, self.l_tri_loss, self.id_criterion = None, None, None

        self.g_tri_loss = TripletLoss(margin=global_margin)

        if self.l_loss_weight > 0:
            self.l_tri_loss = TripletLoss(margin=local_margin)
        if self.id_loss_weight > 0:
            self.id_criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

        self.normalize_feature = normalize_feature

        self.min_val = mindspore.Tensor(1e-12, mindspore.float32)

    def construct(self, logits, labels):
        """ Forward """
        global_feat, local_feat, logits = logits

        g_loss, p_inds, n_inds, _, _, _ = global_loss(
            self.g_tri_loss,
            global_feat,
            labels,
            normalize_feature=self.normalize_feature,
            min_val=self.min_val,
        )

        loss = g_loss * self.g_loss_weight

        if self.l_loss_weight > 0:
            l_loss, _, _ = local_loss(
                self.l_tri_loss,
                local_feat,
                p_inds,
                n_inds,
                normalize_feature=self.normalize_feature,
                min_val=self.min_val,
            )
            loss += l_loss * self.l_loss_weight

        if self.id_loss_weight > 0:
            id_loss = self.id_criterion(logits, labels)  # labels_var
            loss += id_loss * self.id_loss_weight

        return loss
