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
"""
python sdne.py
"""
import mindspore.nn as nn
from tqdm import tqdm
import numpy as np

class SDNE(nn.Cell):
    """
    SDNE

    Args:
        X(Tensor): origin net matrix

    Returns:
        X_(Tensor): net reconstruction matrix
        Y(Tensor): embeddings matrix
    """
    def __init__(self, node_size, hidden_size=None, weight_init='normal'):
        super(SDNE, self).__init__()
        in_channels = node_size
        self.encode = nn.SequentialCell()
        for i in range(len(hidden_size)):
            self.encode.append(nn.Dense(in_channels, hidden_size[i], activation=nn.ReLU(), weight_init=weight_init))
            in_channels = hidden_size[i]

        self.decode = nn.SequentialCell()
        for i in reversed(range(len(hidden_size) - 1)):
            self.decode.append(nn.Dense(in_channels, hidden_size[i], activation=nn.ReLU(), weight_init=weight_init))
            in_channels = hidden_size[i]
        self.decode.append(nn.Dense(in_channels, node_size, activation=nn.ReLU(), weight_init=weight_init))

    def construct(self, X):
        """
        construct
        """
        Y = X
        Y = self.encode(Y)

        X_ = Y.copy()
        X_ = self.decode(X_)

        return X_, Y

class SDNEWithLossCell(nn.Cell):
    """
    SDNEWithLossCell

    Args:
        X(Tensor): origin net matrix
        L(Tensor): laplacian matrix

    Returns:
        loss(float): loss
    """
    def __init__(self, backbone, loss_fn):
        super(SDNEWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, X, L):
        """
        construct
        """
        X_, Y = self._backbone(X)
        return self._loss_fn(X_, Y, X, L)

    def get_backbone(self):
        """
        get backbone
        """
        return self._backbone

    def get_embeddings(self, X):
        """
        get embeddings matrix
        """
        _, embeddings = self._backbone(X)
        return embeddings.asnumpy()

    def get_reconstructions(self, X, idx2node_y, idx2node_x=None):
        """
        get net reconstruction matrix items and their node pairs
        """
        reconstructions, _ = self._backbone(X)
        look_back_x = idx2node_y if idx2node_x is None else idx2node_x
        look_back_y = idx2node_y
        vertices = []
        for i in tqdm(range(reconstructions.shape[0])):
            for j in range(reconstructions.shape[1]):
                vertices.append([look_back_x[i], look_back_y[j]])

        return reconstructions.reshape(-1).asnumpy(), np.array(vertices, dtype=np.int32)
