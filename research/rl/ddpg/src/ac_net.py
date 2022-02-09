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
"""actor-critic net"""
import math
import mindspore.nn as nn
from mindspore.common.initializer import Uniform


class ActorNet(nn.Cell):
    """
        Basic Actor Net
        Args:
            state_dim (int): Input channel.
            action_dim (int): Output channel.
        Returns:
            Tensor, output tensor.
    """
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.a_fc1 = nn.Dense(state_dim, 400, weight_init=Uniform(1 / math.sqrt(state_dim)))
        self.a_fc2 = nn.Dense(400, 300, weight_init=Uniform(1 / math.sqrt(400)))
        self.a_fc3 = nn.Dense(300, action_dim, weight_init=Uniform(3e-3))
        self.a_relu = nn.ReLU()
        self.a_tanh = nn.Tanh()

    def construct(self, x):
        """construct"""
        x = self.a_fc1(x)
        x = self.a_relu(x)
        x = self.a_fc2(x)
        x = self.a_relu(x)
        x = self.a_fc3(x)
        x = self.a_tanh(x)
        return x * 2


class CriticNet(nn.Cell):
    """
        Basic Critic Net
        Args:
            state_dim (int): Input channel.
            action_dim (int): Output channel.
        Returns:
            Tensor, output tensor.
    """
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.c_fc1 = nn.Dense(state_dim, 400, weight_init=Uniform(1 / math.sqrt(state_dim)))
        self.c_fc2 = nn.Dense(action_dim, 300, weight_init=Uniform(1 / math.sqrt(400)))
        self.c_fc3 = nn.Dense(400, 300, weight_init=Uniform(1 / math.sqrt(400)))
        self.c_fc4 = nn.Dense(300, 1, weight_init=Uniform(3e-3))
        self.c_relu = nn.ReLU()

    def construct(self, state, action):
        """construct"""
        state = self.c_fc1(state)
        action = self.c_fc2(action)
        state = self.c_relu(state)
        state = self.c_fc3(state)
        x = state + action
        x = self.c_relu(x)
        x = self.c_fc4(x)
        return x
