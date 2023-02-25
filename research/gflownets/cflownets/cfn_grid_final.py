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
CFlowNet main function; train and test
"""
import random

import numpy as np
import mindspore as ms
from mindspore import context, Tensor, set_seed
from mindspore.common import dtype as mstype
from mindspore import ops

from point_env import MultiStepTwoGoalPointEnv
from transaction import Transaction, Critic
from replay_buffer import ReplayBuffer
from loss_network import TransactionTrainNetWrapper, CriticTrainNetWrapper
from config import CONFIG as cfg
from utils import save_variable, save_model, select_action_base_probability, softmax_matrix

set_seed(cfg['seed'])
random.seed(cfg['seed'])
np.random.seed(cfg['seed'])


class CFN:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, uniform_action_size,
                 discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.transaction = Transaction(state_dim, action_dim)
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.uniform_action_size = uniform_action_size
        self.uniform_action = np.random.uniform(low=-max_action, high=max_action,
                                                size=(uniform_action_size, action_dim))
        self.total_it = 0
        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(self, state, is_max):
        sample_action = np.random.uniform(low=-self.max_action, high=self.max_action, size=(1000, self.action_dim))
        self.critic.set_train(False)
        state = np.repeat(state.reshape(1, -1), 1000, axis=0)
        sa = np.concatenate((state, sample_action), axis=-1)
        sa = Tensor(sa, mstype.float32)
        edge_flow = self.critic(sa).reshape(-1)
        edge_flow = edge_flow.asnumpy()
        edge_flow_norm = np.array(softmax_matrix(edge_flow))
        if is_max == 0:
            action = select_action_base_probability(sample_action, edge_flow_norm)
        elif is_max == 1:
            action = sample_action[edge_flow.argmax()]
        return action

    def cal_inflow_sa(self, next_state, state, action, batch_size, max_episode_steps, sample_flow_num):
        """
        calculate inflow state and action
        """
        uniform_action = np.random.uniform(low=-self.max_action, high=self.max_action,
                                           size=(batch_size, max_episode_steps, sample_flow_num, self.action_dim))
        current_state = np.repeat(next_state, sample_flow_num, axis=2).reshape(
            batch_size, max_episode_steps, sample_flow_num, -1)
        cat_state_action = np.concatenate((current_state, uniform_action), axis=-1)
        cat_state_action = Tensor(cat_state_action, mstype.float32)
        inflow_state = self.transaction(cat_state_action)
        state_ms = Tensor(state.reshape(batch_size, max_episode_steps, -1, self.state_dim), mstype.float32)
        inflow_state = ops.concat([inflow_state, state_ms], axis=2)
        inflow_action = np.concatenate((uniform_action, action.reshape(
            batch_size, max_episode_steps, -1, self.action_dim)), axis=2)
        inflow_action = Tensor(inflow_action, mstype.float32)
        return inflow_state, inflow_action

    def cal_outflow_sa(self, next_state, action, batch_size, max_episode_steps, sample_flow_num):
        """
        calculate outflow state and action
        """
        uniform_action = np.random.uniform(low=-self.max_action, high=self.max_action,
                                           size=(batch_size, max_episode_steps, sample_flow_num, self.action_dim))
        outflow_state = np.repeat(next_state, sample_flow_num + 1, axis=2).reshape(
            batch_size, max_episode_steps, sample_flow_num + 1, -1)
        last_action = np.zeros((batch_size, 1, 1))
        last_action = np.concatenate((action[:, 1:, :], last_action), axis=1)
        last_action_ = last_action.reshape((batch_size, max_episode_steps, -1, self.action_dim))
        outflow_action = np.concatenate((uniform_action, last_action_), axis=2)
        return outflow_state, outflow_action

    def train(self, replay_buffer, critic_train_net, transaction_train_net, frame_idx, done_true):
        sample_flow_num = cfg['sample_flow_num']
        batch_size = cfg['batch_size']
        max_episode_steps = cfg['max_episode_steps']

        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        transaction_train_net.set_train(False)

        # in flow
        inflow_state, inflow_action = self.cal_inflow_sa(next_state, state,
                                                         action, batch_size, max_episode_steps, sample_flow_num)

        # out flow
        outflow_state, outflow_action = self.cal_outflow_sa(next_state,
                                                            action, batch_size, max_episode_steps, sample_flow_num)

        critic_train_net.set_train()
        transaction_train_net.set_train()

        outflow_state = Tensor(outflow_state, mstype.float32)
        outflow_action = Tensor(outflow_action, mstype.float32)
        not_done = Tensor(not_done, mstype.float32)
        done_true = Tensor(done_true, mstype.float32)
        reward = Tensor(reward, mstype.float32)
        next_state = Tensor(next_state, mstype.float32)
        action = Tensor(action, mstype.float32)
        state = Tensor(state, mstype.float32)
        print('frame_idx:', frame_idx)
        critic_loss = critic_train_net(inflow_state, inflow_action, outflow_state, outflow_action, not_done,
                                       done_true, reward)
        print('every critic_loss:', critic_loss)
        transaction_loss = transaction_train_net(next_state, action, state)
        print('every transaction_loss:', transaction_loss)


def main():
    print('device:', cfg['device_platform'])
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_platform'])

    env = MultiStepTwoGoalPointEnv()
    test_env = MultiStepTwoGoalPointEnv()
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --------  define policy --------
    hidden_dim = cfg['hidden_dim']
    uniform_action_size = cfg['uniform_action_size']
    policy = CFN(state_dim, action_dim, hidden_dim, max_action, uniform_action_size)

    # --------  define replay_buffer --------
    replay_buffer_size = cfg['replay_buffer_size']
    replay_buffer = ReplayBuffer(replay_buffer_size)

    #  --------  define train step and loss --------
    critic_train_net = CriticTrainNetWrapper(policy.critic)
    transaction_train_net = TransactionTrainNetWrapper(policy.transaction)

    max_frames = cfg['max_frames']
    start_timesteps = cfg['start_timesteps']
    frame_idx = cfg['frame_idx']
    batch_size = cfg['batch_size']
    test_epoch = cfg['test_epoch']
    repeat_episode_num = cfg['repeat_episode_num']
    sample_episode_num = cfg['sample_episode_num']
    max_episode_steps = cfg['max_episode_steps']

    rewards = []
    done_true = np.zeros((batch_size, max_episode_steps))
    for i in done_true:
        i[max_episode_steps - 1] = 1

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0
        state_buf = []
        action_buf = []
        reward_buf = []
        next_state_buf = []
        done_buf = []

        for step in range(max_episode_steps):
            action = policy.select_action(state, 0)
            next_state, reward, done, _ = env.step(action)
            done_bool = float(1. - done)
            state_buf.append(state)
            action_buf.append(action)
            reward_buf.append(reward)
            next_state_buf.append(next_state)
            done_buf.append(done_bool)
            state = next_state
            episode_reward += reward

            if done:
                frame_idx += 1
                replay_buffer.push(state_buf, action_buf, reward_buf, next_state_buf, done_buf)
                break

            if frame_idx >= start_timesteps and step % 6 == 0:
                policy.train(replay_buffer, critic_train_net, transaction_train_net, frame_idx, done_true)

        if frame_idx > start_timesteps and frame_idx % 25 == 0:
            test_epoch += 1
            avg_test_episode_reward = 0
            for _ in range(repeat_episode_num):
                test_state = test_env.reset()
                test_episode_reward = 0
                for _ in range(max_episode_steps):
                    test_action = policy.select_action(np.array(test_state), 1)
                    test_next_state, test_reward, test_done, _ = test_env.step(test_action)
                    test_state = test_next_state
                    test_episode_reward += test_reward
                    if test_done:
                        break
                avg_test_episode_reward += test_episode_reward
            save_model(policy.critic)
            print('*****************  id:{} ; reward  *****************'.format(frame_idx))
            print(avg_test_episode_reward / repeat_episode_num)

            total_state_buf = []
            total_reward_buf = []
            for _ in range(sample_episode_num):
                test_state = test_env.reset()
                test_state_buf = []
                for _ in range(max_episode_steps):
                    action = policy.select_action(np.array(test_state), 0)
                    next_test_state, reward, done, _ = test_env.step(action)
                    test_state_buf.append(test_state)
                    test_state = next_test_state

                    if done:
                        total_state_buf.append(test_state_buf)
                        total_reward_buf.append(reward)
                        break
            save_variable(total_state_buf, 'data/gfn_MA_state_ceta_' + str(frame_idx) + '_1000.data')
            save_variable(total_reward_buf, 'data/gfn_MA_reward_ceta_' + str(frame_idx) + '_1000.data')
        rewards.append(episode_reward)


def test():
    # config
    max_episode_steps = 12
    hidden_dim = 256
    uniform_action_size = 1000
    max_frames = 5000
    frame_idx = 0

    # --------  define env --------
    env = MultiStepTwoGoalPointEnv()
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --------  define policy --------
    policy = CFN(state_dim, action_dim, hidden_dim, max_action, uniform_action_size)
    param_dict = ms.load_checkpoint("runs/gfn_MA_***.ckpt")
    ms.load_param_into_net(policy.critic, param_dict)

    total_state_buf = []
    total_reward_buf = []
    rewards = []
    dist_buf = []

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0
        state_buf = []
        action_buf = []

        for _ in range(max_episode_steps):
            action = policy.select_action(state, 0)
            next_state, reward, done, dist = env.step(action)

            state_buf.append(state)
            action_buf.append(action)
            state = next_state
            episode_reward += reward

            if done:
                frame_idx += 1
                total_state_buf.append(state_buf)
                total_reward_buf.append(reward)
                dist_buf.append(dist)
                break
        print('*****************  id:{} ; reward  *****************'.format(frame_idx))
        print(episode_reward)
        rewards.append(episode_reward)
    save_variable(total_state_buf, 'gfn_MA_state_ceta5000.data')
    save_variable(total_reward_buf, 'gfn_MA_reward_ceta5000.data')


if __name__ == '__main__':
    main()
