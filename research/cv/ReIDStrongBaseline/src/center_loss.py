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
""" Center Loss and CrossEntropyLabelSmooth"""
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as mnp
from mindspore import ops
from mindspore.common.initializer import initializer, Normal


class CrossEntropyLabelSmooth(nn.Cell):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(axis=1)
        self.exp_dims = ops.ExpandDims()

    def construct(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        m = inputs.shape[0]
        log_probs = self.logsoftmax(inputs)

        classes = mnp.arange(self.num_classes)
        labels = self.exp_dims(targets, 1).repeat(self.num_classes, axis=1)
        classes = self.exp_dims(classes, 0).repeat(m, axis=0)
        targets = (labels == classes).astype(log_probs.dtype)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CenterLoss(nn.Cell):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = Parameter(
            initializer(Normal(sigma=1), [num_classes, feat_dim], mstype.float32),
            name="center",
        )
        self.exp_dims = ops.ExpandDims()
        self.min_val = Tensor(1e-12, mstype.float32)

        self.conc = ops.Concat(axis=1)

    def construct(self, x, labels):
        """ Forward
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """

        m = x.shape[0]
        n = self.num_classes

        xx = ops.pows(x, 2).sum(axis=1, keepdims=True).repeat(n, axis=1)
        yy = ops.pows(self.centers, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
        distmat = xx + yy
        distmat = 1 * distmat - 2 * ops.dot(x, self.centers.transpose())

        classes = mnp.arange(self.num_classes)
        labels = self.exp_dims(labels, 1).repeat(n, axis=1)
        classes = self.exp_dims(classes, 0).repeat(m, axis=0)
        mask = labels == classes
        distmat = distmat * mask

        distmat = ops.maximum(
            distmat,
            self.min_val,
        ) / m
        return distmat.sum()
