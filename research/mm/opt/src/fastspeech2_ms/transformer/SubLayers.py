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
"""fastspeech2 sublayers"""
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np

from .Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Cell):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Dense(d_model, n_head * d_k)
        self.w_ks = nn.Dense(d_model, n_head * d_k)
        self.w_vs = nn.Dense(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm((d_model,))

        self.fc = nn.Dense(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        self.transpose = ops.Transpose()

    def construct(self, q, k, v, mask=None):
        """MultiheadAttention construct"""
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = self.transpose(q, (2, 0, 1, 3)).view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = self.transpose(k, (2, 0, 1, 3)).view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = self.transpose(v, (2, 0, 1, 3)).view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = np.tile(mask, (n_head, 1, 1))  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            self.transpose(output, (1, 2, 0, 3)).view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Cell):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
            pad_mode="pad"
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
            pad_mode="pad"
        )

        self.layer_norm = nn.LayerNorm((d_in,))
        self.dropout = nn.Dropout(dropout)
        self.relu = ops.ReLU()
        self.transpose = ops.Transpose()

    def construct(self, x):
        residual = x
        output = self.transpose(x, (0, 2, 1))
        output = self.w_2(self.relu(self.w_1(output)))
        output = self.transpose(output, (0, 2, 1))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output
