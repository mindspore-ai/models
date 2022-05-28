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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import Zeros, Concat, BroadcastTo
from mindspore.ops import Sin, Cos
from mindspore.numpy import outer
from src.utils.additional_algorithms import linear


class PositionalEmbedding(nn.Cell):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.concat_n_1 = Concat(-1)
        self.sin = Sin()
        self.cos = Cos()
        self.demb = demb
        self.inv_freq = ms.Tensor(1 / (10000 ** (np.arange(0.0, demb, 2.0) / demb)), ms.float32)

    def construct(self, pos_seq, bsz=None):
        sinusoid_inp = outer(pos_seq, self.inv_freq)
        pos_emb = self.concat_n_1([self.sin(sinusoid_inp), self.cos(sinusoid_inp)])

        if bsz is not None:
            return BroadcastTo(-1, bsz, -1)(pos_emb[:, None, :])
        return pos_emb[:, None, :]


class AdaptiveEmbedding(nn.Cell):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1):
        super(AdaptiveEmbedding, self).__init__()
        self.zeros = Zeros()
        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.CellList()
        parameters = []
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed)
            )
            if d_proj != d_embed:
                parameters.append(ms.Parameter(self.zeros((d_proj, d_embed), ms.float32)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                parameters.append(ms.Parameter(self.zeros((d_proj, d_emb_i), ms.float32)))
        self.emb_projs = ms.ParameterTuple(parameters)

    def construct(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = linear(embed, self.emb_projs[0])
        else:
            embed = self.emb_layers[0](inp)

        embed *= self.emb_scale

        return embed
