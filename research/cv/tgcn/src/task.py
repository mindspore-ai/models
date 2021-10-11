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
Supervised forecast task
"""
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from .model.tgcn import TGCN


class SupervisedForecastTask(nn.Cell):
    """
    T-GCN applied to supervised forecast task
    """

    def __init__(self, adj, hidden_dim: int, pre_len: int):
        super(SupervisedForecastTask, self).__init__()
        self.adj = Parameter(Tensor(adj, mstype.float32), name='adj', requires_grad=False)
        self.tgcn = TGCN(self.adj, hidden_dim)
        self.fcn = nn.Dense(hidden_dim, pre_len)

    def construct(self, inputs):
        """
        Calculate network predictions for supervised forecast task

        Args:
            inputs(Tensor): network inputs

        Returns:
            predictions: predictions of supervised forecast task
        """
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = inputs.shape
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.tgcn(inputs)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.shape[2]))
        # (batch_size * num_nodes, pre_len)
        predictions = self.fcn(hidden)
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        # Change data shape for the following calculation of metrics
        predictions = predictions.transpose(0, 2, 1).reshape((-1, num_nodes))
        return predictions
