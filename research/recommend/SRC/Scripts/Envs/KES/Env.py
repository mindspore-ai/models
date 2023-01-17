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
from mindspore import dtype as mdtype, ms_function
from mindspore import ops

from utils import load_d_agent, episode_reward


class KESEnv():
    def __init__(self, dataset, model_name='DKT', dataset_name='assist09'):
        self.skill_num = dataset.feats_num
        self.model = load_d_agent(model_name, dataset_name, self.skill_num)
        self.targets = None
        self.states = (None, None)
        self.initial_score = None

    @ms_function
    def exam(self, targets, states):
        scores = []
        for i in range(targets.shape[1]):
            score, _ = self.model.learn_lstm(targets[:, i:i + 1], *states)  # (B, 1)
            scores.append(score)
        return ops.mean(ops.concat(scores, axis=1), axis=1)  # (B,)

    def begin_episode(self, targets, initial_logs):
        self.targets = targets
        initial_score, initial_log_scores, states = self.begin_episode_(targets, initial_logs)
        self.initial_score = initial_score
        self.states = states
        return initial_log_scores

    @ms_function
    def begin_episode_(self, targets, initial_logs=None):
        states = (None, None)
        score = None
        if initial_logs is not None:
            score, states = self.model.learn_lstm(initial_logs)
        initial_score = self.exam(targets, states)
        return initial_score, score, states

    def n_step(self, learning_path, binary=False):
        scores, states = self.model.learn_lstm(learning_path, *self.states)
        self.states = states
        if binary:
            # scores = ops.bernoulli(scores)
            scores = ops.cast(scores > 0.5, mdtype.float32)
        return scores

    def end_episode(self, **kwargs):
        final_score, reward = self.end_episode_(self.initial_score, self.targets, *self.states)
        if 'score' in kwargs:
            return final_score, reward
        return reward

    @ms_function
    def end_episode_(self, initial_score, targets, states1, states2):
        final_score = self.exam(targets, (states1, states2))
        reward = episode_reward(initial_score, final_score, 1).expand_dims(-1)
        return final_score, reward
