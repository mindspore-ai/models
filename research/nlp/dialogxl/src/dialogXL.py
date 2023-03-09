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
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.common.initializer import initializer, Normal

class XLNetRelativeAttention(nn.Cell):
    def __init__(self, config):
        super(XLNetRelativeAttention, self).__init__()

        if config.d_model % config.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention \
                head (%d)" % (config.d_model, config.n_head)
            )

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = ms.Parameter(initializer(Normal(config.initializer_range),
                                          (config.d_model, self.n_head, self.d_head)), 'q')
        self.k = ms.Parameter(initializer(Normal(config.initializer_range),
                                          (config.d_model, self.n_head, self.d_head)), 'k')
        self.v = ms.Parameter(initializer(Normal(config.initializer_range),
                                          (config.d_model, self.n_head, self.d_head)), 'v')
        self.o = ms.Parameter(initializer(Normal(config.initializer_range),
                                          (config.d_model, self.n_head, self.d_head)), 'o')
        self.r = ms.Parameter(initializer(Normal(config.initializer_range),
                                          (config.d_model, self.n_head, self.d_head)), 'r')

        self.r_r_bias = ms.Parameter(initializer(Normal(config.initializer_range),
                                                 (self.n_head, self.d_head)), 'r_r_bias')
        self.r_w_bias = ms.Parameter(initializer(Normal(config.initializer_range),
                                                 (self.n_head, self.d_head)), 'r_w_bias')

        self.layer_norm = nn.LayerNorm([self.d_model], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.dropout)
        self.softmax = ops.Softmax(-1)

    def rel_shift_bnij(self, x, klen=-1):
        x_size = x.shape

        x = x.view(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.view(x_size[0], x_size[1], x_size[2], x_size[3] - 1)

        x = x[:, :, :, :klen]
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, attn_mask=None):
        # q_head (qlen, B, H, d_h)
        # k_head_h (klen, B, H, d_h)
        # v_head_h (qlen, B, H, d_h)
        # k_head_r (qlen, B, H, d_h)
        # content based attention score
        ac = ops.matmul((q_head + self.r_w_bias).transpose((1, 2, 0, 3)), k_head_h.transpose((1, 2, 3, 0)))

        # position based attention score
        bd = ops.matmul((q_head + self.r_r_bias).transpose((1, 2, 0, 3)), k_head_r.transpose((1, 2, 3, 0)))
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # merge attention scores and perform masking
        attn_score = (ac + bd) * self.scale
        if attn_mask is not None:
            attn_score = attn_score - 1e30 * attn_mask.transpose((2, 3, 0, 1))

        # attention probability
        attn_prob = self.softmax(attn_score)
        attn_prob = self.dropout(attn_prob)

        # attention output
        attn_vec = ops.matmul(attn_prob, v_head_h.transpose((1, 2, 0, 3))).transpose((2, 0, 1, 3))

        return attn_vec

    def post_attention(self, h, attn_vec):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        # h(qlen, B, D)
        # attn_vec (qlen, B, H, d_h)
        # o(D, H, d_h)
        attn_out = ops.tensor_dot(attn_vec, self.o, ((2, 3), (1, 2)))
        attn_out = self.dropout(attn_out)
        attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def construct(self, h, attn_mask_h, r, mems):
        # Multi_head attention with relative positional encoding
        if mems is not None:
            cat = mnp.concatenate((mems, h))
        else:
            cat = h

        q_head_h = ops.tensor_dot(h, self.q, ((2,), (0,)))
        k_head_h = ops.tensor_dot(cat, self.k, ((2,), (0,)))
        v_head_h = ops.tensor_dot(cat, self.v, ((2,), (0,)))
        k_head_r = ops.tensor_dot(r, self.r, ((2,), (0,)))

        # core attention ops
        attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, attn_mask_h)

        # post processing
        output = self.post_attention(h, attn_vec)

        return output

class XLNetFeedForward(nn.Cell):
    def __init__(self, config):
        super(XLNetFeedForward, self).__init__()

        self.layer_norm = nn.LayerNorm([config.d_model], epsilon=config.layer_norm_eps)
        self.layer_1 = nn.Dense(config.d_model, config.d_inner)
        self.layer_2 = nn.Dense(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        if config.ff_activation == 'gelu':
            self.activation_function = nn.GELU()
        elif config.ff_activation == 'relu':
            self.activation_function = nn.ReLU()

    def construct(self, inp):
        output = self.layer_1(inp)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output

class XLNetLayer(nn.Cell):
    def __init__(self, config):
        super(XLNetLayer, self).__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(p=config.dropout)

    def construct(self, input_h, attn_mask_h, r, mems):
        output_h = self.rel_attn(input_h, attn_mask_h, r, mems)
        output_h = self.ff(output_h)
        return output_h

class DialogXL(nn.Cell):
    def __init__(self, config):
        super(DialogXL, self).__init__()
        self.mem_len = config.mem_len
        self.d_model = config.d_model
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.windowp = config.windowp
        self.num_heads = config.num_heads

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layer = nn.CellList([XLNetLayer(config) for _ in range(self.n_layer)])
        self.dropout = nn.Dropout(p=config.dropout)

    def create_mask(self, qlen, speaker_mask, window_mask, speaker_ids):
        if speaker_mask is not None:
            mlen, bsz = speaker_mask.shape

            speaker_ids = speaker_ids.expand_dims(0).expand_dims(0)
            attn_mask = mnp.zeros((qlen, qlen, bsz, self.n_head))

            speaker_mask_expand = speaker_mask.expand_dims(0)
            local_mask_pad = (window_mask.expand_dims(0) <= 0)
            global_mask_pad = (speaker_mask_expand == 0)
            speaker_mask_pad = ops.logical_or(global_mask_pad, speaker_mask_expand != speaker_ids)
            listener_mask_pad = ops.logical_or(global_mask_pad, speaker_mask_expand == speaker_ids)

            local_mask_pad = ops.broadcast_to(local_mask_pad[:, :, :, None],
                                              (qlen, mlen, bsz, self.num_heads[0])).astype(ms.float32)
            global_mask_pad = ops.broadcast_to(global_mask_pad[:, :, :, None],
                                               (qlen, mlen, bsz, self.num_heads[1])).astype(ms.float32)
            speaker_mask_pad = ops.broadcast_to(speaker_mask_pad[:, :, :, None],
                                                (qlen, mlen, bsz, self.num_heads[2])).astype(ms.float32)
            listener_mask_pad = ops.broadcast_to(listener_mask_pad[:, :, :, None],
                                                 (qlen, mlen, bsz, self.num_heads[3])).astype(ms.float32)

            mask_pad = mnp.concatenate((local_mask_pad, global_mask_pad, speaker_mask_pad, listener_mask_pad), axis=3)

            ret = mnp.concatenate((mask_pad, attn_mask), axis=1)
            return ret
        return None

    def cache_mem(self, curr_out, prev_mem, content_lengths):
        if prev_mem is None:
            new_mem = mnp.zeros((self.mem_len, len(content_lengths), curr_out.shape[2]))
            for i, length in enumerate(content_lengths):
                new_mem[-length+1:, i] = curr_out[1:length, i]
        else:
            mems = []
            for i, length in enumerate(content_lengths):
                if length > 1:
                    mems.append(mnp.concatenate((prev_mem[:, i],
                                                 curr_out[1:length, i]))[-self.mem_len:].view(self.mem_len,
                                                                                              self.d_model))
                else:
                    mems.append(prev_mem[:, i])
            new_mem = mnp.stack(mems)
            new_mem = new_mem.transpose((1, 0, 2))
        return ops.stop_gradient(new_mem)

    def cache_speaker_mask(self, prev_speaker_mask, speaker_ids, content_lengths, content_mask):
        if prev_speaker_mask is None:
            new_speaker_mask = mnp.zeros((self.mem_len, len(content_lengths)))
            for i, length in enumerate(content_lengths):
                new_speaker_mask[-length+1:, i] = content_mask[1:length, i] * speaker_ids[i]
        else:
            masks = []
            for i, length in enumerate(content_lengths):
                if length > 1:
                    masks.append(
                        mnp.concatenate(
                            (prev_speaker_mask[:, i],
                             content_mask[1:length, i] * speaker_ids[i]))[-self.mem_len:].view(self.mem_len))
                else:
                    masks.append(prev_speaker_mask[:, i])
            new_speaker_mask = mnp.stack(masks)              # (B, mlen)
            new_speaker_mask = new_speaker_mask.transpose((1, 0))
        return ops.stop_gradient(new_speaker_mask)

    def cache_window_mask(self, prev_window_mask, content_lengths, content_mask):
        if prev_window_mask is None:
            new_window_mask = mnp.zeros((self.mem_len, len(content_lengths)))
            for i, length in enumerate(content_lengths):
                new_window_mask[-length+1:, i] = content_mask[1:length, i] * self.windowp
        else:
            masks = []
            for i, length in enumerate(content_lengths):
                if length > 1:
                    masks.append(
                        mnp.concatenate(
                            (prev_window_mask[:, i] - 1,
                             content_mask[1:length, i] * self.windowp))[-self.mem_len:].view(self.mem_len))
                else:
                    masks.append(prev_window_mask[:, i] - 1)
            new_window_mask = mnp.stack(masks)              # (B, mlen)
            new_window_mask = new_window_mask.transpose((1, 0))  # (mlen, B)
        return ops.stop_gradient(new_window_mask)

    def positional_embedding(self, pos_seq, inv_freq, bsz=None):
        sinusoid_inp = ops.matmul(pos_seq.expand_dims(-1), inv_freq.expand_dims(0))
        pos_emb = mnp.concatenate((mnp.sin(sinusoid_inp), mnp.cos(sinusoid_inp)), axis=-1)
        pos_emb = pos_emb[:, None, :]

        pos_emb = ops.broadcast_to(pos_emb, (pos_emb.shape[0], bsz, pos_emb.shape[2]))
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        freq_seq = mnp.arange(0, self.d_model, 2, dtype=mnp.float32)
        inv_freq = 1 / mnp.power(10000, (freq_seq / self.d_model))

        beg, end = klen, -qlen
        fwd_pos_seq = mnp.arange(beg, end, -1, dtype=mnp.float32)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz) #(klen, B, D)
        return pos_emb

    def construct(self, input_ids, mems, content_lengths, content_mask,
                  speaker_ids, speaker_mask, window_mask):
        input_ids = input_ids.transpose((1, 0))
        qlen, bsz = input_ids.shape[0], input_ids.shape[1]

        content_mask = content_mask.transpose((1, 0))

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        if speaker_mask is not None:
            attn_mask = self.create_mask(qlen, speaker_mask, window_mask, speaker_ids)
        else:
            attn_mask = None

        word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)

        pos_emb = self.relative_positional_encoding(qlen, klen, bsz)
        pos_emb = self.dropout(pos_emb)

        if mems is None:
            mems = [None] * self.n_layer

        new_speaker_mask = self.cache_speaker_mask(speaker_mask, speaker_ids, content_lengths, content_mask)
        new_window_mask = self.cache_window_mask(window_mask, content_lengths, content_mask)

        new_mems = []
        for i, layer_module in enumerate(self.layer):
            new_mems.append(self.cache_mem(output_h, mems[i], content_lengths))
            output_h = layer_module(output_h, attn_mask, pos_emb, mems[i])
        new_mems = mnp.stack(new_mems)

        output_h = self.dropout(output_h)
        output_h = output_h.transpose((1, 0, 2))
        return output_h, new_mems, new_speaker_mask, new_window_mask

class ERC_xlnet(nn.Cell):
    def __init__(self, args, backbone):
        super(ERC_xlnet, self).__init__()
        self.dropout = nn.Dropout(p=args.output_dropout)
        self.xlnet = backbone

        self.pool_fc = nn.SequentialCell([nn.Dense(args.bert_dim, args.hidden_dim, weight_init='he_uniform'),
                                          nn.ReLU()])

        layers = []
        for _ in range(args.mlp_layers):
            layers += [nn.Dense(args.hidden_dim, args.hidden_dim, weight_init='he_uniform'), nn.ReLU()]
        layers += [nn.Dense(args.hidden_dim, args.n_classes, weight_init='he_uniform')]
        self.out_mlp = nn.SequentialCell(layers)

    def construct(self, content_ids, mems, content_mask, content_lengths, speaker_ids,
                  speaker_mask, window_mask):
        text_features, new_mems, \
        new_speaker_mask, new_window_mask = self.xlnet(content_ids, mems, content_lengths, content_mask,
                                                       speaker_ids, speaker_mask, window_mask)
        text_out = self.pool_fc(text_features[:, 0, :])
        text_out = self.dropout(text_out)
        outputs = self.out_mlp(text_out)
        return outputs, new_mems, new_speaker_mask, new_window_mask
