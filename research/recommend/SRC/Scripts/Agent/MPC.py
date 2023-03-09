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
from mindspore import nn, numpy as mnp, ops, ms_function

from KTScripts.BackModels import MLP


class MPC(nn.Cell):
    def __init__(self, skill_num, input_size, hidden_size, pre_hidden_sizes, dropout, hor):
        super(MPC).__init__()
        self.l1 = nn.SequentialCell(nn.Dense(input_size + 1, input_size),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=1 - dropout))
        self.embed = nn.Embedding(skill_num, input_size)
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = MLP(hidden_size, pre_hidden_sizes + [1], dropout=dropout, norm_layer=None)
        self.hor = hor

    @ms_function
    def sample(self, b, n):
        candidate_order = mnp.rand(b, n)
        candidate_order = ops.sort(candidate_order)[1]  # (B*Hor, n)
        return candidate_order

    @ms_function
    def test(self, targets, states):  # (B, H) or (B*Hor, H)
        x, _ = self.encoder(targets, states)
        x = x[:, -1]
        x = ops.sigmoid(self.decoder(x).squeeze())
        return x

    def begin_episode(self, targets, initial_logs, initial_log_scores):
        # targets: (B, K), where K is the num of targets in this batch
        targets = mnp.mean(self.embed(targets), axis=1, keepdims=True)  # (B, 1, I)
        targets_repeat = targets.repeat(self.hor, 0)  # (B*Hor, 1, I)
        if initial_logs is not None:
            states = self.step(initial_logs, initial_log_scores, None)
        else:
            zeros = ops.zeros_like(targets).swapaxes(0, 1)  # (1, B, Hor)
            states = (zeros, zeros)
        return targets, targets_repeat, states

    def step(self, x, score, states):
        x = self.embed(x)
        if score is not None:
            x = self.l1(ops.concat((x, score.expand_dims(-1)), -1))
        _, states = self.encoder(x, states)
        return states

    def construct(self, targets, initial_logs, initial_log_scores, origin_path, n):
        targets, targets_repeat, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        unselected = mnp.ones_like(origin_path, dtype=mnp.bool_)
        a1 = mnp.arange(targets.shape[0])
        a2 = a1
        a1 = a1.expand_dims(1).repeat(n, 1).repeat(self.hor, 0)  # (B*Hor, n)
        b1 = mnp.arange(unselected.shape[0]).expand_dims(1).repeat(unselected.shape[1], 1)
        b2 = mnp.arange(unselected.shape[1]).expand_dims(0).repeat(unselected.shape[0], 0)
        result_path = []
        target_args = None
        max_len, batch = origin_path.shape[1], targets_repeat.shape[0]
        for i in range(n):
            candidate_args = self.sample(batch, max_len - i)[:, :(n - i)]  # (B*H, n-i)
            if i > 0:
                candidate_args = candidate_args.view(-1, self.hor, n - i)
                candidate_args[:, -1] = target_args
                candidate_args = candidate_args.view(-1, n - i)
            candidate_paths = ops.masked_select(origin_path, unselected)
            candidate_paths = candidate_paths.view(-1, max_len - i)[a1, candidate_args]  # (B*Hor, n-i)
            a1 = a1[:, :-1]
            states_repeat = [_.repeat(self.hor, 1) for _ in states]
            _, states_repeat = self.encoder(self.embed(candidate_paths), states_repeat)  # (B*Hor, L, H)
            candidate_scores = self.test(targets_repeat, states_repeat).view(-1, self.hor)
            selected_hor = mnp.argmax(candidate_scores, axis=1)  # (B,)

            target_args = candidate_args.view(-1, self.hor, n - i)[a2, selected_hor]
            target_path = candidate_paths.view(-1, self.hor, n - i)[a2, selected_hor]
            result_path.append(target_path[:, 0])

            modified = ops.masked_select(unselected, unselected).view(unselected.shape[0], -1)
            modified[a2, target_args[:, 0]] = False
            temp1, temp2 = ops.masked_select(b1, unselected), ops.masked_select(b2, unselected)
            unselected[temp1, temp2] = modified.view(temp1.shape)
            target_args = mnp.where(
                mnp.greater(target_args, target_args[:, :1]), target_args - 1, target_args)
            target_args = target_args[:, 1:]

            states = self.step(target_path[:, :1], None, states)
        result_path = mnp.stack(result_path, axis=1)
        return result_path

    @ms_function
    def backup(self, targets, initial_logs, initial_log_scores, result_path):
        _, _, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        history_states, _ = self.encoder(self.embed(result_path), states)
        history_scores = ops.sigmoid(self.decoder(history_states)).squeeze(-1)  # (B, L)
        return history_scores
