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
""" ReID Strong Baseline losses """

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops

from src.center_loss import CenterLoss, CrossEntropyLabelSmooth
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
    def get_max(dist_mat__, idxs, inv=False):
        """ Search max values in distance matrix (min if inv=True) """
        dist_mat_ = dist_mat__.copy()
        if inv:
            dist_mat_ = -dist_mat_
        # fill distances for non-idxs values as min value
        dist_mat_[~idxs] = dist_mat_.min() - 1
        pos_max = dist_mat_.argmax(axis=-1)
        maxes = dist_mat__.take(pos_max, axis=-1).diagonal()
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


class StrongLoss(nn.Cell):
    """ Combined loss for ReID Strong Baseline model

    Args:
        num_classes: number of classes
        center_loss_weight: weight of Center loss
        crossentropy_loss_weight: weight of CE loss
        feat_dim: number of features for Center loss
        margin: Triplet loss margin
    """
    def __init__(self, num_classes, center_loss_weight=0.0005, crossentropy_loss_weight=1, feat_dim=2048, margin=0.3):
        super().__init__()
        self.center_loss_weight = center_loss_weight
        self.crossentropy_loss_weight = crossentropy_loss_weight

        self.triplet = TripletLoss(margin=margin)
        self.min_val = Tensor(1e-12, mstype.float32)
        self.xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)

    def construct(self, logits, labels):
        """ Forward """
        score, feat = logits
        target = labels
        tloss = global_loss(self.triplet, feat, target, self.min_val)[0]
        xloss = self.xent(score, target)
        closs = self.center_criterion(feat, target)

        loss = self.crossentropy_loss_weight * xloss + tloss + self.center_loss_weight * closs

        return loss
