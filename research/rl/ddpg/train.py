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
"""Ascend910train"""

import os
import argparse
import gym
import numpy as np
from mindspore import save_checkpoint
from mindspore import context
import src.agent
from src.config import config

MEMORY_CAPACITY = config.MEMORY_CAPACITY
EPISODES = config.EPISODES
EP_STEPS = config.EP_STEPS


parsers = argparse.ArgumentParser(description='MindSpore ddpg Example')
parsers.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                     help='device where the code will be implemented (default: Ascend)')
parsers.add_argument('--device_id', type=int, default=0, help='if is test, must provide\
                    path where the trained ckpt file')
args = parsers.parse_args()

context.set_context(device_id=args.device_id)


def train():
    """ train"""
    env = gym.make('Pendulum-v0')
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    a_low_bound = env.action_space.low
    ddpg_agent = src.agent.Agent(a_dim, s_dim, a_bound)
    var = 3
    for i in range(EPISODES):
        state = env.reset()
        ep_r = 0
        for j in range(EP_STEPS):
            action = ddpg_agent.choose_action(state)
            action = action.asnumpy()
            action = np.clip(np.random.normal(action, var), a_low_bound, a_bound)
            next_state, reward, _, _ = env.step(action)
            ddpg_agent.store_transition(state, action, reward / 10, next_state)
            if ddpg_agent.point > MEMORY_CAPACITY:
                var *= 0.9995
                ddpg_agent.learn()
            state = next_state
            ep_r += reward
            if j == EP_STEPS - 1:
                print('Episode: ', i, ' Reward: %i' % ep_r, 'Explore: %.2f' % var)

    save_checkpoint(ddpg_agent.actor_net, os.getcwd() + "/../actor_net.ckpt")
    save_checkpoint(ddpg_agent.actor_target, os.getcwd() + "/../actor_target.ckpt")
    save_checkpoint(ddpg_agent.critic_net, os.getcwd() + "/../critic_net.ckpt")
    save_checkpoint(ddpg_agent.critic_target, os.getcwd() + "/../critic_target.ckpt")


if __name__ == '__main__':
    train()
