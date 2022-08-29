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

import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import Tril, Triu, Dense
from mindspore.ops import Zeros, Ones
from mindspore.ops import ExpandDims, Concat, Split
from mindspore.ops import Transpose, Tile
from mindspore.ops import Softmax

from src.common.ac import AcCell
from src.common.attnveccell import AttnVecCell
from src.common.bd import BdCell
from src.utils.additional_algorithms import MaskerFill


class RelMultiHeadAttn(nn.Cell):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.0, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.zeros, self.ones = Zeros(), Ones()
        self.expandDims, self.concat_0, self.concat_1 = ExpandDims(), Concat(0), Concat(1)
        self.split_n_1_2, self.split_n_1_3 = Split(-1, 2), Split(-1, 3)
        self.tril, self.triu = Tril(), Triu()
        self.transpose = Transpose()
        self.tile = Tile()
        self.maskerFill = MaskerFill()
        self.softmax_1 = Softmax(1)

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = Dense(d_model, 3 * n_head * d_head, has_bias=False).to_float(ms.float16)

        self.drop = nn.Dropout(1 - dropout, dtype=ms.float32)
        self.dropatt = nn.Dropout(1 - dropout, dtype=ms.float32)

        self.o_net = Dense(n_head * d_head, d_model, has_bias=False).to_float(ms.float16)

        self.layer_norm = nn.LayerNorm([d_model])

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        self.negative_inf = -1e9

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = self.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), x.dtype)
        x_padded = self.concat_1((zero_pad, x))

        x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], x.shape[2], x.shape[3])

        x = x_padded[1:].reshape(x.shape)

        if zero_triu:
            _ones = self.ones((x.shape[0], x.shape[1]))
            x = x * self.tril(_ones, x.shape[1] - x.shape[0])[:, :, None, None]

        return x

    def construct(self, w, r, r_w_bias, r_r_bias, mems=None, attn_mask=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = Dense(self.d_model, self.n_head * self.d_head, has_bias=False).to_float(ms.float16)
        self.ac_cell = AcCell().to_float(ms.float16)
        self.bd_cell = BdCell().to_float(ms.float16)
        self.AttnVecCell = AttnVecCell().to_float(ms.float16)

    def construct(self, w, r, r_w_bias, r_r_bias, mems=None, attn_mask=None):
        qlen, rlen, bsz = w.shape[0], r.shape[0], w.shape[1]

        # if mems is not None and mems.ndim > 1:
        if not self.is_first_iteration:
            cat = self.concat_0([mems, w])
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = self.split_n_1_3(w_heads)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = self.split_n_1_3(w_heads)

        klen = w_head_k.shape[0]

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        rr_head_q = w_head_q + r_r_bias

        rw_head_q_fp16 = self.cast(rw_head_q, ms.float16)
        w_head_k_fp16 = self.cast(w_head_k, ms.float16)
        AC = self.ac_cell(rw_head_q_fp16, w_head_k_fp16)
        AC = self.cast(AC, ms.float32)

        rr_head_q_fp16 = self.cast(rr_head_q, ms.float16)
        r_head_k_fp16 = self.cast(r_head_k, ms.float16)
        BD = self.bd_cell(rr_head_q_fp16, r_head_k_fp16)
        BD = self.cast(BD, ms.float32)

        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD

        attn_score *= self.scale

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask_ = self.tile(self.expandDims(self.expandDims(attn_mask, -1), 0),
                                       (1, 1, attn_score.shape[2], attn_score.shape[3]))
                attn_score = self.maskerFill(attn_score, attn_mask_, self.negative_inf)
            elif attn_mask.ndim == 3:
                attn_mask_ = self.tile(self.expandDims(attn_mask, -1), (1, 1, attn_score.shape[2], attn_score.shape[3]))
                attn_score = self.maskerFill(attn_score, attn_mask_, self.negative_inf)

        # [qlen x klen x bsz x n_head]
        attn_prob = self.softmax_1(attn_score)
        attn_prob = self.dropatt(attn_prob)
        # compute attention vector

        attn_prob_fp16 = self.cast(attn_prob, ms.float16)
        w_head_v_fp16 = self.cast(w_head_v, ms.float16)
        attn_vec = self.AttnVecCell(attn_prob_fp16, w_head_v_fp16)
        attn_vec = self.cast(attn_vec, ms.float32)
        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.reshape(
            attn_vec.shape[0], attn_vec.shape[1], self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output
