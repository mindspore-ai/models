# Copyright 2023 Huawei Technologies Co., Ltd
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
network
"""
import mindspore.nn as nn
from mindspore.ops import operations as P


class Transaction(nn.Cell):
    def __init__(self, state_dim, action_dim):
        super(Transaction, self).__init__()
        self.t_l1 = nn.Dense(state_dim + action_dim, 256, has_bias=True, weight_init='Uniform', bias_init='Uniform')
        self.t_l2 = nn.Dense(256, 256, has_bias=True, weight_init='Uniform', bias_init='Uniform')
        self.t_l3 = nn.Dense(256, 256, has_bias=True, weight_init='Uniform', bias_init='Uniform')
        self.t_l4 = nn.Dense(256, state_dim, has_bias=True, weight_init='Uniform', bias_init='Uniform')
        self.cat_axis0 = P.Concat(axis=0)
        self.t_relu = nn.ReLU()

    def construct(self, sa):
        q1 = self.t_relu(self.t_l1(sa))
        q1 = self.t_relu(self.t_l2(q1))
        q1 = self.t_relu(self.t_l3(q1))
        q1 = self.t_l4(q1)
        return q1


class Critic(nn.Cell):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        # Edge flow network architecture
        self.c_l1 = nn.Dense(state_dim + action_dim, hidden_dim, has_bias=True, weight_init='Uniform',
                             bias_init='Uniform')
        self.c_l2 = nn.Dense(hidden_dim, hidden_dim, has_bias=True, weight_init='Uniform', bias_init='Uniform')
        self.c_l3 = nn.Dense(hidden_dim, 1, has_bias=True, weight_init='Uniform', bias_init='Uniform')
        self.c_relu = nn.ReLU()
        self.softplus = P.Softplus()

    def construct(self, sa):
        q1 = self.c_relu(self.c_l1(sa))
        q1 = self.c_relu(self.c_l2(q1))
        q1 = self.softplus(self.c_l3(q1))
        return q1
