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
"""
python loss.py
"""
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops

class SDNELoss1(nn.Cell):
    """
    SDNELoss1

    Args:
        X_(Tensor): net reconstruction matrix
        Y(Tensor): embeddings matrix
        X(Tensor): origin net matrix
        L(Tensor): laplacian matrix

    Returns:
        loss(float): loss
    """
    def __init__(self, alpha=1, beta=1, gamma=1):
        super(SDNELoss1, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.matmul1 = ops.MatMul()
        self.matmul2 = ops.MatMul(transpose_a=True)

    def construct(self, X_, Y, X, L):
        """
        construct
        """
        loss1 = self._loss_1st(Y, L)
        loss2 = self._loss_2nd(X_, X)
        return self.gamma * loss1 + self.alpha * loss2

    def _loss_2nd(self, X_, X):
        """
        2nd loss
        """
        B = np.ones_like(X_)
        B[X_ != 0] = self.beta
        loss = np.square((X - X_) * B)
        loss = np.sum(loss, axis=-1)
        return np.mean(loss)

    def _loss_1st(self, Y, L):
        """
        1st loss
        """
        batch_size = L.shape[0]
        loss = self.matmul2(Y, L)
        loss = self.matmul1(loss, Y)
        return 2 * loss.trace() / batch_size

class SDNELoss2(nn.Cell):
    """
    SDNELoss1

    Args:
        X_(Tensor): net reconstruction matrix
        Y(Tensor): embeddings matrix
        X(Tensor): origin net matrix
        Xadj(Tensor): adjacency matrix corresponding to X

    Returns:
        loss(float): loss
    """
    def __init__(self, alpha=1, beta=1, gamma=1):
        super(SDNELoss2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.matmul1 = ops.MatMul()
        self.matmul2 = ops.MatMul(transpose_a=True)

    def construct(self, X_, Y, X, Xadj):
        """
        construct
        """
        loss1 = self._loss_1st(Y, Xadj)
        loss2 = self._loss_2nd(X_, X)

        return self.gamma * loss1 + self.alpha * loss2

    def _loss_2nd(self, X_, X):
        """
        2nd loss
        """
        B = X * (self.beta - 1) + 1
        loss = np.square((X_ - X) * B)

        return np.sum(loss)

    def _loss_1st(self, Y, Xadj):
        """
        1st loss
        """
        D = np.diag(np.sum(Xadj, 1))
        L = D - Xadj
        loss = self.matmul2(Y, L)
        loss = self.matmul1(loss, Y)

        return 2 * loss.trace()
