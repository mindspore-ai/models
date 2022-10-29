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
"""calculate QMIX loss and update model parameters"""

import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore_rl.agent import Learner
from mindspore_rl.utils import SoftUpdate


class LossCell(nn.Cell):

    def __init__(self, params, agents, mixer, target_mixer):
        super().__init__()
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.agents = agents
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.gatherd = ops.GatherD()

    def construct(self, obs_batch, current_state_batch, next_state_batch,
                  action_batch, avail_action_batch, reward_batch, mask,
                  filled_batch, q_val_target, hidden_state_batch):
        reshaped_local_obs = obs_batch.reshape((-1, obs_batch.shape[-1]))
        reshaped_hidden_states = hidden_state_batch.reshape((-1, 64))
        # q values of current state selected actions
        q_val, _ = self.agents(reshaped_local_obs, reshaped_hidden_states)
        q_val = q_val.reshape(
            (self.batch_size, avail_action_batch.shape[1],
             avail_action_batch.shape[2], avail_action_batch.shape[3]))
        choosen_q_val = self.gatherd(q_val, 3, action_batch)
        choosen_q_val = choosen_q_val[:, :-1].squeeze(-1)
        # max q values for the next state
        next_q_val = q_val[:, 1:]
        next_q_val[avail_action_batch[:, 1:] == 0] = -1e10
        max_q_val = next_q_val.argmax(axis=-1).expand_dims(-1)
        max_q_val = self.gatherd(q_val_target, 3, max_q_val).squeeze(-1)
        # TD target
        q_tot_policy = self.mixer(choosen_q_val, current_state_batch)
        q_tot_target = self.target_mixer(max_q_val, next_state_batch)
        target = reward_batch + self.gamma * mask * q_tot_target
        loss = ops.square(
            (q_tot_policy - target) * filled_batch).sum() / filled_batch.sum()
        return loss


class QMIXLearner(Learner):

    def __init__(self, params):
        super().__init__()
        self.agents = params['agents']
        self.target_agents = self.agents.clone()
        self.mixer = params['mixer']
        self.target_mixer = self.mixer.clone()
        self.batch_size = params['batch_size']

        train_params = self.agents.trainable_params(
        ) + self.mixer.trainable_params()
        target_params = self.target_agents.trainable_params(
        ) + self.target_mixer.trainable_params()
        self.target_soft_updater = SoftUpdate(1, 200, train_params,
                                              target_params)

        optimizer = nn.RMSProp(train_params, learning_rate=params['lr'])
        qmix_loss_cell = LossCell(params, self.agents, self.mixer,
                                  self.target_mixer)
        self.train_step = nn.TrainOneStepCell(qmix_loss_cell, optimizer)
        self.train_step.set_train(mode=True)

    def learn(self, experience):
        obs_batch, state_batch, action_batch, reward_batch, done_batch,\
        hidden_state_batch, avail_action_batch, filled_batch = experience
        # drop the last time step
        current_state_batch = state_batch[:, :-1]
        next_state_batch = state_batch[:, 1:]
        reward_batch = reward_batch[:, :-1]
        done_batch = done_batch[:, :-1]
        filled_batch = filled_batch[:, :-1]

        transposed_obs_batch = obs_batch.transpose((1, 0, 2, 3))
        target_hidden_state = ops.zeros(
            (obs_batch.shape[2] * self.batch_size, 64), ms.float32)
        q_val_target = []
        episode_len = obs_batch.shape[1]
        # make learning target for each step
        for i in range(episode_len):
            step_obs = transposed_obs_batch[i]
            obs = step_obs.reshape(step_obs.shape[0] * step_obs.shape[1], -1)
            step_target_q_val, target_hidden_state = self.target_agents(
                obs, target_hidden_state)
            step_target_q_val = step_target_q_val.reshape(
                (-1, avail_action_batch.shape[-2],
                 avail_action_batch.shape[-1]))
            q_val_target.append(step_target_q_val)
        mask = 1 - done_batch
        q_val_target = ops.stack(q_val_target, axis=1)[:, 1:]
        q_val_target[avail_action_batch[:, 1:] == 0] = -1e10
        # calculate loss
        loss = self.train_step(obs_batch, current_state_batch,
                               next_state_batch, action_batch,
                               avail_action_batch, reward_batch, mask,
                               filled_batch, q_val_target, hidden_state_batch)
        # update target network parameters
        self.target_soft_updater()
        return loss
