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

import mindspore.numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Parameter, Tensor
from mindspore.ops import Zeros, Ones, clip_by_value
from mindspore.ops import ExpandDims, Concat
from mindspore.nn import Tril, Triu, Dense

from src.loss_fn.ProjectedAdaptiveLogSoftmaxLoss import ProjectedAdaptiveLogSoftmaxLoss
from src.model.embedding import AdaptiveEmbedding, PositionalEmbedding
from src.model.layer import RelPartialLearnableDecoderLayer


class MemTransformerLM(nn.Cell):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, batch_size, d_embed=None,
                 div_val=1, pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, eval_tgt_len=None,
                 cutoffs=None, sample_softmax=-1, tie_weight=True, tie_projs=None,
                 same_length=False, clamp_len=-1):
        super(MemTransformerLM, self).__init__()

        if tie_projs is None:
            tie_projs = [False]
        if cutoffs is None:
            cutoffs = []
        self.assign = P.Assign()
        self.zeros, self.ones = Zeros(), Ones()
        self.expandDims, self.concat_0, self.concat_1 = ExpandDims(), Concat(0), Concat(1)
        self.tril, self.triu = Tril(), Triu()

        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.batch_size = batch_size

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(1 - dropout, dtype=ms.float32)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

        self.eval_tgt_len = eval_tgt_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.layers = nn.CellList()

        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if self.sample_softmax > 0:
            self.out_layer = Dense(d_model, n_token).to_float(ms.float16)
            if tie_weight:
                self.out_layer.weight = self.word_emb.emb_projs[0].embedding_table
            self.tie_weight = tie_weight

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmaxLoss(n_token, d_embed, d_model,
                                                        cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].embedding_table

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = Tensor(clamp_len, ms.float32)
        self.min_clamp_len = Tensor(0, ms.float32)

        self._create_params()

        self.idx = 0

        self.add_flags_recursive(is_first_iteration=True)

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = Parameter(self.zeros((self.n_head, self.d_head), ms.float32))
        self.r_r_bias = Parameter(self.zeros((self.n_head, self.d_head), ms.float32))
        self.mems = Parameter(
            self.zeros((self.n_layer, self.mem_len, self.batch_size, self.d_model), ms.float32),
            requires_grad=False)
        self.empty_valid_mems = self.zeros(
            (self.n_layer, self.mem_len + self.tgt_len - self.eval_tgt_len, self.batch_size, self.d_model), ms.float32)
        self.valid_mems = Parameter(self.empty_valid_mems, requires_grad=False)

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        return True

    def _update_mems(self, hids, qlen, mlen):
        if self.training:  # update mems #
            if self.mem_len > 0:
                # There are `mlen + qlen` steps that can be cached into mems
                # For the next step, the last `ext_len` of the `qlen` tokens
                # will be used as the extended context. Hence, we only cache
                # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
                # to `mlen + qlen - self.ext_len`.
                for i, h in enumerate(hids):
                    hids[i] = self.expandDims(h, 0)

                # graph mode not support function max()
                end_idx = mlen if qlen - self.ext_len < 0 else qlen - self.ext_len + mlen
                beg_idx = 0 if end_idx - self.mem_len < 0 else end_idx - self.mem_len
                cat = self.concat_0(hids)
                cat = self.concat_1((self.mems, cat))
                cat = cat[:, beg_idx:end_idx]
                self.assign(self.mems, cat)
        else:  # update mems #
            if self.mem_len > 0:
                for i, h in enumerate(hids):
                    hids[i] = self.expandDims(h, 0)

                if self.is_first_iteration:
                    cat = self.concat_0(hids)
                    cat = self.sameShape(cat, self.valid_mems)
                    self.assign(self.valid_mems, cat)
                else:
                    end_idx = mlen if qlen - self.ext_len < 0 else qlen - self.ext_len + mlen
                    beg_idx = 0 if end_idx - self.mem_len < 0 else end_idx - self.mem_len
                    cat = self.concat_0(hids)
                    cat = self.concat_1((self.valid_mems, cat))
                    cat = cat[:, beg_idx:end_idx]
                    self.assign(self.valid_mems, cat)
        return True

    def sameShape(self, a, b):
        c = self.zeros((a.shape[0], b.shape[1] - a.shape[1], a.shape[2], a.shape[3]), ms.float32)
        a = self.concat_1((c, a))
        return a

    def set_train(self, tgt_len, ext_len, mem_len, eval_tgt_len, mode=True):
        super(MemTransformerLM, self).set_train(mode=mode)
        if mode:
            # Switch back to the training mode
            self.reset_length(tgt_len, ext_len, mem_len)
        else:
            # If the model does not use memory at all, make the ext_len longer.
            # Otherwise, make the mem_len longer and keep the ext_len the same.
            self.add_flags_recursive(is_first_iteration=True)
            self.assign(self.valid_mems, self.empty_valid_mems)
            if mem_len == 0:
                self.reset_length(eval_tgt_len,
                                  ext_len + tgt_len - eval_tgt_len, mem_len)
            else:
                self.reset_length(eval_tgt_len,
                                  ext_len, mem_len + tgt_len - eval_tgt_len)
        return True

    def construct(self, data, target, idx=None):
        tgt_len = target.size
        qlen, _ = data.shape
        word_emb = self.word_emb(data)

        mems = self.mems if self.training else self.valid_mems
        mlen = 0 if self.is_first_iteration \
            else (self.mem_len if self.training else self.mem_len + self.tgt_len - self.eval_tgt_len)

        klen = qlen + mlen
        all_ones = np.ones((qlen, klen), ms.int32)

        if self.same_length:
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = np.expand_dims((np.triu((all_ones, 1 + mlen), ms.int32)
                                            + np.tril((all_ones, -mask_shift_len), ms.int32)), -1)  # -1
        else:
            dec_attn_mask = np.expand_dims(np.triu(all_ones, 1 + mlen), -1)

        hids = []

        pos_seq = np.arange(klen - 1, -1, -1, dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq = clip_by_value(pos_seq, clip_value_min=self.min_clamp_len, clip_value_max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        for i, layer in enumerate(self.layers):
            hids.append(core_out)
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, attn_mask=dec_attn_mask, mems=mems[i])

        hidden = self.drop(core_out)

        self._update_mems(hids, qlen, mlen)

        ###########################################################################
        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.reshape(-1, pred_hid.shape[-1]), target.reshape(-1))

        return loss
