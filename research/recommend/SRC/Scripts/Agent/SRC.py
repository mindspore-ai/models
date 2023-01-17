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
from mindspore import nn, ops, ms_function, numpy as mnp, dtype

from KTScripts.BackModels import MLP, Transformer


class SRC(nn.Cell):
    def __init__(self, skill_num, input_size, weight_size, hidden_size, dropout, allow_repeat=False,
                 with_kt=False):
        super(SRC).__init__()
        self.embedding = nn.Embedding(skill_num, input_size)
        self.l1 = nn.Dense(input_size + 1, input_size)
        self.l2 = nn.Dense(input_size, hidden_size)
        self.state_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.path_encoder = Transformer(hidden_size, hidden_size, 0.0, head=1, b=1, transformer_mask=False)
        self.W1 = nn.Dense(hidden_size, weight_size, has_bias=False)  # blending encoder
        self.W2 = nn.Dense(hidden_size, weight_size, has_bias=False)  # blending decoder
        self.vt = nn.Dense(weight_size, 1, has_bias=False)  # scaling sum of enc and dec by v.T
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        if with_kt:
            self.ktRnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.ktMlp = MLP(hidden_size, [hidden_size // 2, hidden_size // 4, 1], dropout=dropout)
        self.allow_repeat = allow_repeat
        self.withKt = with_kt
        self.skill_num = skill_num

    @ms_function
    def begin_episode(self, targets, initial_logs, initial_log_scores):
        # targets: (B, K), where K is the num of targets in this batch
        targets = self.l2(ops.mean(self.embedding(targets), axis=1, keep_dims=True))  # (B, 1, H)
        if initial_logs is not None:
            states = self.step(initial_logs, initial_log_scores, None)
        else:
            zeros = ops.zeros_like(targets).swapaxes(0, 1)  # (1, B, H)
            states = (zeros, zeros)
        return targets, states

    def step(self, x, score, states):
        x = self.embedding(x)
        x = self.l1(ops.concat((x, score.expand_dims(-1)), -1))
        _, states = self.state_encoder(x, states)
        return states

    @ms_function
    def construct(self, targets, initial_logs, initial_log_scores, origin_path, n):
        targets, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        inputs = self.l2(self.embedding(origin_path))
        encoder_states = inputs
        encoder_states = self.path_encoder(encoder_states)
        encoder_states += inputs
        blend1 = self.W1(encoder_states + ops.mean(encoder_states, axis=1, keep_dims=True) + targets)  # (B, L, W)
        decoder_input = mnp.zeros_like(inputs[:, 0:1])  # (B, 1, I)
        probs, paths = [], []
        selecting_s = []
        a1 = mnp.arange(inputs.shape[0])
        selected = mnp.zeros_like(inputs[:, :, 0], dtype=dtype.bool_)
        minimum_fill = mnp.full_like(selected, -1e9, dtype=dtype.float32)
        hidden_states = []
        for i in range(n):
            hidden, states = self.decoder(decoder_input, states)
            if self.withKt and i > 0:
                hidden_states.append(hidden)
            # Compute blended representation at each decoder time step
            blend2 = self.W2(hidden)  # (B, 1, W)
            blend_sum = blend1 + blend2  # (B, L, W)
            out = self.vt(blend_sum).squeeze(-1)  # (B, L)
            if not self.allow_repeat:
                out = mnp.where(selected, minimum_fill, out)
                out = ops.softmax(out, axis=-1)
                if self.training:
                    selecting = ops.multinomial(out, 1).squeeze(-1)
                else:
                    selecting = mnp.argmax(out, 1)
                selected[a1, selecting] = True
            else:
                out = ops.softmax(out, axis=-1)
                selecting = ops.multinomial(out, 1).squeeze(-1)
            selecting_s.append(selecting)
            path = origin_path[a1, selecting]
            decoder_input = encoder_states[a1, selecting].expand_dims(1)
            out = out[a1, selecting]
            paths.append(path)
            probs.append(out)
        probs = ops.stack(probs, 1)
        paths = ops.stack(paths, 1)  # (B, n)
        selecting_s = ops.stack(selecting_s, 1)
        if self.withKt and self.training:
            hidden_states.append(self.decoder(decoder_input, states)[0])
            hidden_states = ops.concat(hidden_states, axis=1)
            kt_output = ops.sigmoid(self.ktMlp(hidden_states))
            result = [paths, probs, selecting_s, kt_output]
            return result
        return paths, probs, selecting_s

    @ms_function
    def backup(self, targets, initial_logs, initial_log_scores, origin_path, selecting_s):
        targets, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        inputs = self.l2(self.embedding(origin_path))
        encoder_states = inputs
        encoder_states = self.path_encoder(encoder_states)
        encoder_states += inputs
        blend1 = self.W1(encoder_states + ops.mean(encoder_states, axis=1, keep_dims=True) + targets)  # (B, L, W)
        selecting_states = encoder_states[mnp.arange(encoder_states.shape[0]).expand_dims(1), selecting_s]
        selecting_states = ops.concat((mnp.zeros_like(selecting_states[:, 0:1]), selecting_states[:, :-1]), 1)
        hidden_states, _ = self.decoder(selecting_states, states)
        blend2 = self.W2(hidden_states)  # (B, n, W)
        blend_sum = blend1.expand_dims(1) + blend2.expand_dims(2)  # (B, n, L, W)
        out = self.vt(blend_sum).squeeze(-1)  # (B, n, L)
        # Masking probabilities according to output order
        mask = mnp.expand_dims(selecting_s, 1).repeat(selecting_s.shape[-1], 1)  # (B, n, n)
        mask = mnp.tril(mask + 1, -1).view(-1, mask.shape[-1])
        out = out.view(-1, out.shape[-1])
        out = ops.concat((ops.zeros_like(out[:, 0:1]), out), -1)
        out[mnp.arange(out.shape[0]).expand_dims(1), mask] = -1e9
        out = out[:, 1:].view(origin_path.shape[0], -1, origin_path.shape[1])

        out = ops.softmax(out, axis=-1)
        probs = ops.gather_elements(out, 2, selecting_s.expand_dims(-1)).squeeze(-1)
        return probs
