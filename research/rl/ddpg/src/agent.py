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
"""agent"""
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import src.ac_net
import src.loss_net
from src.config import config


MEMORY_CAPACITY = config.MEMORY_CAPACITY
BATCH_SIZE = config.BATCH_SIZE
TAU = config.TAU
GAMMA = config.GAMMA
LR_ACTOR = config.LR_ACTOR
LR_CRITIC = config.LR_CRITIC
CRITIC_DECAY = config.CRITIC_DECAY


class Agent:
    """
        Agent Net
        Args:
            state_dim (int): Input channel.
            action_dim (int): Output channel.
            action_bound(float): the bound of action
    """
    def __init__(self, action_dim, state_dim, action_bound):
        self.action_dim, self.state_dim, self.action_bound = action_dim, state_dim, action_bound
        self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + self.action_dim + 1), dtype=np.float32)
        self.point = 0

        self.actor_net = src.ac_net.ActorNet(state_dim, action_dim)
        self.actor_target = src.ac_net.ActorNet(state_dim, action_dim)
        self.critic_net = src.ac_net.CriticNet(state_dim, action_dim)
        self.critic_target = src.ac_net.CriticNet(state_dim, action_dim)

        self.actor_optimizer = nn.Adam(self.actor_net.trainable_params(), learning_rate=LR_ACTOR)
        self.critic_optimizer = nn.Adam(self.critic_net.trainable_params(), learning_rate=LR_CRITIC,
                                        weight_decay=CRITIC_DECAY)

        self.loss_func = nn.MSELoss()

        self.loss_actor_net = src.loss_net.ActorWithLossCell(actor_net=self.actor_net, critic_net=self.critic_net)
        self.loss_critic_net = src.loss_net.CriticWithLossCell(critic_network=self.critic_net,
                                                               loss_func=self.loss_func)

        self.train_actor_net = nn.TrainOneStepCell(self.loss_actor_net, self.actor_optimizer)
        self.train_critic_net = nn.TrainOneStepCell(self.loss_critic_net, self.critic_optimizer)
        self.train_critic_net.set_train(mode=True)
        self.train_actor_net.set_train(mode=True)

    def store_transition(self, state, action, reward, next_state):
        """
             store transition into memory
             Args:
                 state (numpy): state of gym
                 action (numpy): action of gym
                 reward(float): reward of gym
                 next_state(numpy) new state of gym
        """
        transition = np.hstack((state, action, reward, next_state))
        index = self.point % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.point += 1

    def choose_action(self, states):
        """
             choose actions from actor net
             Args:
                 states (numpy): state of gym
             Returns:
                 action (Tensor): action of gym
        """
        states = Tensor(states, mindspore.float32)
        expand_dims = ops.ExpandDims()
        states = expand_dims(states, 0)
        action = self.actor_net(states)
        action = action.asnumpy()
        action = np.asarray(action)
        action = action[np.argmax(action)]
        action = Tensor(action, mindspore.float32)
        action.requires_gard = False
        return action

    def soft_update_paras(self, eval_weights, target_weights):
        """
             choose actions from actor net
             Args:
                 eval_weights (Parameters): the parameters of net
                 target_weights(Parameters): the parameters of net
        """
        assign = ops.Assign()
        for i in range(len(eval_weights)):
            t = 0.01 * eval_weights[i] + 0.99 * target_weights[i]
            t = ops.stop_gradient(t)
            assign(target_weights[i], t)
        return self

    def learn(self):
        """train method"""
        self.soft_update_paras(self.actor_net.trainable_params(), self.actor_target.trainable_params())
        self.soft_update_paras(self.critic_net.trainable_params(), self.critic_target.trainable_params())

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        batch_s = Tensor(batch_trans[:, :self.state_dim], mindspore.float32)
        batch_a = Tensor(batch_trans[:, self.state_dim:self.state_dim + self.action_dim], mindspore.float32)
        batch_r = Tensor(batch_trans[:, -self.state_dim - 1: -self.state_dim], mindspore.float32)
        batch_s_ = Tensor(batch_trans[:, -self.state_dim:], mindspore.float32)

        self.train_actor_net(batch_s)

        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        self.train_critic_net(batch_s, batch_a, q_target)
