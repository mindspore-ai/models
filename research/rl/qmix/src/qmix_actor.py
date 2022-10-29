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
"""network structure and QMIX policy implementation"""

import copy
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.numpy as np
import mindspore.nn.probability.distribution as msd
from mindspore.common.api import cells_compile_cache
from mindspore_rl.agent import Actor
from mindspore_rl.agent import trainer


class RNNAgent(nn.Cell):

    def __init__(self, params):
        super(RNNAgent, self).__init__()
        self.hidden_dim = params["hidden_dim"]
        rnn_type = params["rnn_type"]
        self.rnn_layer_num = params["rnn_layer_num"]
        # state_space_dim and action_space_dim is automatically initiated in MSRL
        self.input_dim = params["state_space_dim"] + params[
            "environment_config"]["num_agent"] + params["action_space_dim"]
        self.output_dim = params["action_space_dim"]
        self.fc1 = nn.Dense(self.input_dim, self.hidden_dim, activation='relu')
        rnn = getattr(nn, rnn_type)
        self.rnn = rnn(self.hidden_dim, self.hidden_dim, self.rnn_layer_num)
        self.fc2 = nn.Dense(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        return Tensor(np.zeros((self.rnn_layer_num, self.hidden_dim)),
                      dtype=self.fc1.weights.dtype)

    def construct(self, obs, hidden_states):
        x = self.fc1(obs).expand_dims(0)
        hidden_states = hidden_states.expand_dims(0)
        h, _ = self.rnn(x, hidden_states)
        h = h.squeeze(0)
        q = self.fc2(h)
        return q, h

    def clone(self):
        new_obj = copy.deepcopy(self)
        cells_compile_cache[id(new_obj)] = new_obj.compile_cache
        return new_obj


class QMixNet(nn.Cell):

    def __init__(self, params):
        super(QMixNet, self).__init__()
        self.agent_num = params["environment_config"]["num_agent"]
        self.state_dim = params['environment_config']['global_observation_dim']
        self.embed_dim = params["embed_dim"]
        # hyper-network for state-dependent weight and bias
        self.hyper_weight = nn.Dense(self.state_dim,
                                     self.embed_dim * self.agent_num)
        self.hyper_bias = nn.Dense(self.state_dim, self.embed_dim)
        self.hyper_weight_final = nn.Dense(self.state_dim, self.embed_dim)
        self.bmm = ops.BatchMatMul()
        self.elu = nn.ELU()
        # value estimate as bias for the second layer of the mixing network
        self.value_net = nn.SequentialCell([
            nn.Dense(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dense(self.embed_dim, 1)
        ])

    def construct(self, agent_q, states):
        batch_size = agent_q.shape[0]
        states = states.reshape((-1, self.state_dim))
        agent_q = agent_q.reshape((-1, 1, self.agent_num))
        # first layer of the mixing network
        w1 = ops.abs(self.hyper_weight(states))
        b1 = self.hyper_bias(states)
        weight1 = w1.reshape((-1, self.agent_num, self.embed_dim))
        bias1 = b1.reshape((-1, 1, self.embed_dim))
        hidden = self.elu(self.bmm(agent_q, weight1) + bias1)
        # second layer of the mixing network
        w2 = ops.abs(self.hyper_weight_final(states))
        weight2 = w2.reshape((-1, self.embed_dim, 1))
        value = self.value_net(states).reshape((-1, 1, 1))
        q_tot = (self.bmm(hidden, weight2) + value).reshape(
            (batch_size, -1, 1))
        return q_tot

    def clone(self):
        new_obj = copy.deepcopy(self)
        cells_compile_cache[id(new_obj)] = new_obj.compile_cache
        return new_obj


class QMIXPolicy:

    class CollectPolicy(nn.Cell):
        """The collect policy implementation (how to obtain actions)"""

        def __init__(self, network, params):
            super().__init__()
            self.agent_num = params["environment_config"]["num_agent"]
            self.epsi_start = Tensor([params['epsi_start']], ms.float32)
            self.epsi_end = Tensor([params['epsi_end']], ms.float32)
            all_steps = params['all_steps']
            self.delta = (self.epsi_start - self.epsi_end) / all_steps
            self.network = network
            self.categorical_dist = msd.Categorical()
            self.rand = ops.UniformReal()

        def construct(self, params, step):
            # epsilon decay
            epsilon = ops.maximum(self.epsi_start - self.delta * step,
                                  self.epsi_end)
            obs, hidden_state, avail_action = params
            q_vals, hidden_state = self.network(obs, hidden_state)
            # soft mask non-available actions
            q_vals[avail_action == 0] = -1e10
            # epsilon-greedy action selection
            best_action = self.categorical_dist.mode(q_vals)
            random_action = self.categorical_dist.sample((), avail_action)
            cond = self.rand(
                (self.agent_num, 1)).reshape([self.agent_num]) < epsilon
            selected_action = self.categorical_dist.select(
                cond, random_action, best_action)
            selected_action = selected_action.expand_dims(1)
            return selected_action, hidden_state

    class EvalPolicy(nn.Cell):

        def __init__(self, network):
            super().__init__()
            self.network = network
            self.categorical_dist = msd.Categorical()

        def construct(self, params):
            obs, hidden_state, avail_action = params
            q_vals, hidden_state = self.network(obs, hidden_state)
            q_vals[avail_action == 0] = -1e10
            # greedy action selection
            best_action = self.categorical_dist.mode(q_vals)
            selected_action = best_action.expand_dims(1)
            return selected_action, hidden_state

    def __init__(self, params):
        self.agents = RNNAgent(params)
        self.mixer = QMixNet(params)
        self.collect_policy = self.CollectPolicy(self.agents, params)
        self.eval_policy = self.EvalPolicy(self.agents)


class QMIXActor(Actor):

    def __init__(self, params):
        super().__init__()
        self.collect_policy = params['collect_policy']
        self.eval_policy = params['eval_policy']
        self.collect_env = params['collect_environment']
        self.eval_env = params['eval_environment']

    def get_action(self, phase, params):
        # get action from policy
        obs, hidden_state, avail_action, step = params
        if phase in (trainer.INIT, trainer.COLLECT):
            action, hidden_state = self.collect_policy(
                (obs, hidden_state, avail_action), step)
        elif phase == trainer.EVAL:
            action, hidden_state = self.eval_policy(
                (obs, hidden_state, avail_action))
        else:
            raise Exception("Invalid phase")
        return action, hidden_state

    def act(self, phase, params):
        # select policy to execute in the corresponding environment
        obs, hidden_state, avail_action, step = params
        if phase in (trainer.INIT, trainer.COLLECT):
            action, hidden_state = self.collect_policy(
                (obs, hidden_state, avail_action), step)
            next_obs, reward, done, next_state, next_avail_action = self.collect_env.step(
                action)
        elif phase == trainer.EVAL:
            action, hidden_state = self.eval_policy(
                (obs, hidden_state, avail_action))
            next_obs, reward, done, next_state, next_avail_action = self.eval_env.step(
                action)
        else:
            raise Exception("Invalid phase")
        result = (next_obs, done, reward, action, hidden_state, next_state,
                  next_avail_action)
        return result
