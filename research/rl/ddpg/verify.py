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
"""Ascend910verify"""
import os
import argparse
import gym
import src.ac_net
import src.agent
from src.config import config
from mindspore import load_checkpoint
from mindspore import context


parser = argparse.ArgumentParser(description='MindSpore ddpg Example')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--device_id', type=int, default=0, help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()

context.set_context(device_id=args.device_id)
EP_TEST = config.EP_TEST
STEP_TEST = config.STEP_TEST
REWORD_SCOPE = 16.2736044


def verify():
    """ verify"""
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    verify_agent = src.agent.Agent(action_dim, state_dim, action_bound)
    load_checkpoint(os.getcwd()+"/../actor_net.ckpt", net=verify_agent.actor_net)
    load_checkpoint(os.getcwd()+"/../actor_target.ckpt", net=verify_agent.actor_target)
    load_checkpoint(os.getcwd()+"/../critic_net.ckpt", net=verify_agent.critic_net)
    load_checkpoint(os.getcwd()+"/../critic_target.ckpt", net=verify_agent.critic_target)
    rewards = []
    for i in range(EP_TEST):
        reward_sum = 0
        state = env.reset()
        for j in range(STEP_TEST):
            action = verify_agent.choose_action(state)
            action = action.asnumpy()
            next_state, reward, _, _ = env.step(action)
            reward_sum += reward
            state = next_state
            if j == STEP_TEST - 1:
                print('Episode: ', i, ' Reward:', reward_sum / REWORD_SCOPE)
                rewards.append(reward_sum)
                break
    print('Final Average Reward: ', sum(rewards) / (len(rewards) * REWORD_SCOPE))


if __name__ == '__main__':
    verify()
