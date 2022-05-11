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
python sdne.py
"""
import mindspore.nn as nn
import mindspore.numpy as np

class SDNE(nn.Cell):
    """
    SDNE

    Args:
        X(Tensor): origin net matrix

    Returns:
        X_(Tensor): net reconstruction matrix
        Y(Tensor): embeddings matrix
    """
    def __init__(self, node_size, hidden_size=None, act='relu', weight_init='normal'):
        super(SDNE, self).__init__()
        in_channels = node_size
        self.encode = nn.SequentialCell()
        activation = nn.Sigmoid() if act == 'sigmoid' else nn.ReLU()
        for i in range(len(hidden_size)):
            self.encode.append(nn.Dense(in_channels, hidden_size[i], activation=activation, weight_init=weight_init))
            in_channels = hidden_size[i]

        self.decode = nn.SequentialCell()
        for i in reversed(range(len(hidden_size) - 1)):
            self.decode.append(nn.Dense(in_channels, hidden_size[i], activation=activation, weight_init=weight_init))
            in_channels = hidden_size[i]
        self.decode.append(nn.Dense(in_channels, node_size, activation=activation, weight_init=weight_init))

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
        Xadj(Tensor): adjacency matrix corresponding to X

    Returns:
        loss(float): loss
    """
    def __init__(self, backbone, loss_fn):
        super(SDNEWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, X, Xadj):
        """
        construct
        """
        X_, Y = self._backbone(X)
        loss = self._loss_fn(X_, Y, X, Xadj)

        return loss

    def get_backbone(self):
        """
        get backbone
        """
        return self._backbone

    def get_reconstructions(self, X, idx2node):
        """
        get net reconstruction matrix items and their node pairs
        """
        reconstructions, _ = self._backbone(X)
        look_back = idx2node
        vertices = []
        for i in range(reconstructions.shape[0]):
            for j in range(reconstructions.shape[1]):
                vertices.append([look_back[i], look_back[j]])

        return reconstructions.reshape(-1).asnumpy(), np.array(vertices, dtype=np.int32).asnumpy()

    def get_embeddings(self, X, batch=256):
        """
        get embeddings matrix
        """
        nodenum = X.shape[0]
        if batch <= 0:
            batch = nodenum
        _, embeddings = self._backbone(X[0: min(batch, nodenum)])
        cnt = embeddings.shape[0]
        while cnt < nodenum:
            _, tmp = self._backbone(X[cnt: min(cnt + batch, nodenum)])
            embeddings = np.vstack((embeddings, tmp))
            cnt += batch

        return embeddings.asnumpy()
