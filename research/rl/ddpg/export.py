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
"""export"""

import math
import gym
import mindspore as ms
from mindspore import Tensor, export, context, nn, ops
from mindspore import load_checkpoint
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


def run_export():
    """export"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    state = env.reset()
    state = Tensor(state, ms.float32)
    expand_dims = ops.ExpandDims()
    state = expand_dims(state, 0)
    actor_net = ActorNet(state_dim, action_dim)
    load_checkpoint("actor_net.ckpt", net=actor_net)
    export(actor_net, state, file_name="test", file_format="MINDIR")
    print("export MINDIR file at {}".format("./test.mindir"))

if __name__ == '__main__':
    run_export()
