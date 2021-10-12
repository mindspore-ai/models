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
"""DGCN Network."""
import numpy as np
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.nn.layer.activation import get_activation


def glorot(shape):
    """Randomly generated weight."""
    W = np.asarray(
        np.random.RandomState(1234).uniform(
            low=-np.sqrt(6. / (shape[0]+shape[1])),
            high=np.sqrt(6. / (shape[0]+shape[1])),
            size=(shape[0], shape[1])
        ), dtype=np.float32)
    return Tensor(W)


class GraphConvolution(nn.Cell):
    """Graph convolutional layer."""
    def __init__(self,
                 feature_in_dim,
                 feature_out_dim,
                 dropout_ratio=None,
                 activation=None,
                 ):
        super(GraphConvolution, self).__init__()
        self.in_dim = feature_in_dim
        self.out_dim = feature_out_dim
        self.weight_init = glorot([self.out_dim, self.in_dim])
        self.fc = nn.Dense(self.in_dim,
                           self.out_dim,
                           weight_init=self.weight_init,
                           has_bias=False)
        self.dropout_flag = False
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio is not None:
            self.dropout_flag = self.dropout_ratio
            self.dropout = nn.Dropout(keep_prob=1-self.dropout_ratio)
        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.matmul = P.MatMul()

    def construct(self, adj, input_feature):
        """Convolutional operations."""
        dropout = input_feature
        if self.dropout_flag:
            dropout = self.dropout(dropout)

        fc = self.fc(dropout)
        output_feature = self.matmul(adj, fc)

        if self.activation_flag:
            output_feature = self.activation(output_feature)
        return output_feature


class DGCN(nn.Cell):
    """Generate DGCN model."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(DGCN, self).__init__()
        self.layer0 = GraphConvolution(input_dim, hidden_dim, activation='relu', dropout_ratio=dropout)
        self.layer1 = GraphConvolution(hidden_dim, output_dim, dropout_ratio=dropout)

    def construct(self, adj, ppmi, feature):
        Softmax = nn.Softmax()
        diffoutput0 = self.layer0(adj, feature)
        diffoutput1 = Softmax(self.layer1(adj, diffoutput0))
        ppmioutput0 = self.layer0(ppmi, feature)
        ppmioutput1 = Softmax(self.layer1(ppmi, ppmioutput0))
        return diffoutput1, ppmioutput1
