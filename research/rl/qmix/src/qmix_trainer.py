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
"""QMIX trainer to interact with the environment and replay buffer"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.api import ms_function
from mindspore_rl.agent import Trainer, trainer


class QMIXTrainer(Trainer):

    def __init__(self, msrl, params):
        nn.Cell.__init__(self, auto_prefix=False)
        self.msrl = msrl
        self.params = params
        self.hidden_dim = self.params["hidden_dim"]
        self.batch_size = self.params["batch_size"]
        self.env = self.msrl.collect_environment
        self.eval_env = self.msrl.eval_environment
        self.agent_num = self.env.config['num_agent']
        self.agent_id = Tensor(
            np.expand_dims(np.eye(self.agent_num),
                           0).reshape(self.agent_num, -1), ms.float32)
        self.episode_limit = self.env.config['episode_limit']

        # environment input dimension
        self.action_dim = self.env.action_space.num_values
        self.observation_dim = (self.env.observation_space.shape[-1] +
                                self.agent_num + self.action_dim)
        self.state_dim = self.env.config['global_observation_dim']
        self.reward_dim = self.env.reward_space.shape[-1]
        self.done_dim = self.env.done_space.shape[-1]
        self.step = Parameter(Tensor(0, ms.int32), requires_grad=False)
        self.epsilon_steps = Parameter(Tensor(0, ms.int32),
                                       requires_grad=False,
                                       name='epsilon_steps')
        # operator init
        self.onehot = ops.OneHot()
        self.greateq = ops.GreaterEqual()
        super(QMIXTrainer, self).__init__(msrl)

    def trainable_variables(self):
        trainable_variables = {
            "agents": self.msrl.learner.agents,
            "mixer": self.msrl.learner.mixer
        }
        return trainable_variables

    @ms_function
    def init_episode_buffer(self):
        size = self.episode_limit + 1
        buffer = {
            "obs":
            ops.zeros((size, self.agent_num, self.observation_dim),
                      ms.float32),
            "state":
            ops.zeros((size, self.state_dim), ms.float32),
            "action":
            ops.zeros((size, self.agent_num, 1), ms.int32),
            "reward":
            ops.zeros((size, self.reward_dim), ms.float32),
            "done":
            ops.zeros((size, self.done_dim), ms.bool_),
            "hidden_state":
            ops.zeros((size, self.agent_num, 64), ms.float32),
            "avail_action":
            ops.zeros((size, self.agent_num, self.action_dim), ms.int32),
            "filled":
            ops.zeros((size, self.done_dim), ms.int32)
        }
        return buffer

    @ms_function
    def train_one_episode(self):
        done = Tensor([False], ms.bool_)
        episode_step = Tensor(0, ms.int32)
        reward_sum = Tensor(0, ms.float32)
        loss = Tensor(0, ms.float32)

        episode_buffer = self.init_episode_buffer()
        obs, state, avail_action = self.env.reset()
        onehot_action = ops.zeros((self.agent_num, self.action_dim),
                                  ms.float32).reshape((self.agent_num, -1))
        concat_obs = ops.concat((obs, onehot_action, self.agent_id), axis=1)
        hidden_state = ops.zeros((self.agent_num, self.hidden_dim), ms.float32)
        episode_buffer["obs"][episode_step] = concat_obs
        episode_buffer["state"][episode_step] = state
        episode_buffer["avail_action"][episode_step] = avail_action
        episode_step += 1
        while (not done) and (episode_step < self.episode_limit):
            next_obs, done, reward, action, hidden_state, \
            next_state, next_avail_action = self.msrl.agent_act(
                trainer.COLLECT,
                (concat_obs, hidden_state, avail_action, self.step))
            # set new state variable
            obs = next_obs
            avail_action = next_avail_action
            if ops.equal(self.episode_limit, episode_step).expand_dims(axis=0):
                # traj ends if reaches episode limit
                done = Tensor(False, ms.bool_)
            reward_sum += reward.squeeze(axis=0)
            onehot_action = self.onehot(action, self.action_dim,
                                        Tensor(1, ms.float32),
                                        Tensor(0, ms.float32)).astype(
                                            ms.float32).reshape(
                                                (self.agent_num, -1))
            concat_obs = ops.concat((obs, onehot_action, self.agent_id),
                                    axis=1)
            # store transition
            episode_buffer["obs"][episode_step] = concat_obs
            episode_buffer["state"][episode_step] = next_state
            episode_buffer["action"][episode_step - 1] = action
            episode_buffer["reward"][episode_step - 1] = reward
            episode_buffer["done"][episode_step - 1] = done
            episode_buffer["hidden_state"][episode_step] = hidden_state
            episode_buffer["avail_action"][episode_step] = avail_action
            episode_buffer["filled"][episode_step - 1] = Tensor(
                1, ms.int32).expand_dims(axis=0)
            episode_step += 1

        action, hidden_state = self.msrl.agent_get_action(
            trainer.COLLECT,
            (concat_obs, hidden_state, avail_action, self.step))
        onehot_action = self.onehot(action, self.action_dim,
                                    Tensor(1, ms.float32),
                                    Tensor(0, ms.float32)).reshape(
                                        (self.agent_num, -1))
        concat_obs = ops.concat((obs, onehot_action, self.agent_id), axis=1)
        episode_buffer["obs"][episode_step] = concat_obs
        episode_buffer["action"][episode_step - 1] = action
        episode_buffer["hidden_state"][episode_step] = hidden_state
        self.step += episode_step
        self.msrl.replay_buffer_insert(episode_buffer.values())
        if self.greateq(self.msrl.buffers.count, self.batch_size):
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
        step_info = self.env.get_step_info()
        info = (loss, reward_sum, episode_step, step_info)
        return info

    @ms_function
    def evaluate(self):
        """Evaluation function"""
        done = Tensor([False], ms.bool_)
        obs, _, avail_action = self.eval_env.reset()
        onehot_action = ops.zeros((self.agent_num, self.action_dim),
                                  ms.float32).reshape((self.agent_num, -1))
        concat_obs = ops.concat((obs, onehot_action, self.agent_id), axis=1)
        hidden_state = ops.zeros((self.agent_num, self.hidden_dim), ms.float32)
        while not done:
            next_obs, done, _, action, hidden_state, _, next_avail_action = self.msrl.agent_act(
                trainer.EVAL,
                (concat_obs, hidden_state, avail_action, self.step))
            onehot_action = self.onehot(action, self.action_dim,
                                        Tensor(1, ms.float32),
                                        Tensor(0, ms.float32)).reshape(
                                            (self.agent_num, -1))
            # set new state variables
            obs = next_obs
            avail_action = next_avail_action
            concat_obs = ops.concat((obs, onehot_action, self.agent_id),
                                    axis=1)
        step_info = self.eval_env.get_step_info()
        return step_info
