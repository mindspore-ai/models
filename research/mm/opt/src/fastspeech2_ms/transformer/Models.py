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
"""fastspeech2"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

import fastspeech2_ms.transformer.Constants as Constants
from fastspeech2_ms.text.symbols import symbols
from .Layers import FFTBlock



def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return Tensor(sinusoid_table, dtype=ms.float32)


class Encoder(nn.Cell):
    """ Encoder """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.expand_dims = ops.ExpandDims()
        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = ms.Parameter(
            self.expand_dims(get_sinusoid_encoding_table(n_position, d_word_vec), 0),
            requires_grad=False,
        )

        self.layer_stack = nn.CellList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.cast = ops.Cast()

    def construct(self, src_seq, mask, return_attns=False):
        """Encoder construct"""
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        batch_size, mask_size = mask.shape[0], mask.shape[1]

        # -- Prepare masks
        broadcast_to = ops.BroadcastTo((batch_size, max_len, mask_size))
        slf_attn_mask = broadcast_to(self.expand_dims(self.cast(mask, ms.int32), 1))
        slf_attn_mask = self.cast(slf_attn_mask, ms.bool_)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            table = get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)[: src_seq.shape[1], :]
            table_dim0, table_dim1 = table.shape[0], table.shape[1]
            broadcast_to = ops.BroadcastTo((batch_size, table_dim0, table_dim1))
            enc_output = self.src_word_emb(src_seq) + broadcast_to(self.expand_dims(table, 0))
        else:
            position_enc = self.position_enc[:, :max_len, :]
            pos_dim0, pos_dim1 = position_enc.shape[1], position_enc.shape[2]
            broadcast_to = ops.BroadcastTo((batch_size, pos_dim0, pos_dim1))
            enc_output = self.src_word_emb(src_seq) + broadcast_to(position_enc)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Cell):
    """ Decoder """
    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.expand_dims = ops.ExpandDims()
        self.position_enc = ms.Parameter(
            self.expand_dims(get_sinusoid_encoding_table(n_position, d_word_vec), 0),
            requires_grad=False,
        )

        self.layer_stack = nn.CellList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def construct(self, enc_seq, mask, return_attns=False):
        """Decoder construct"""
        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks

            mask_dim0, mask_dim1 = mask.shape[0], mask.shape[1]
            slf_attn_mask = self.expand_dims(mask, 1)
            broadcast_to = ops.BroadcastTo((mask_dim0, max_len, mask_dim1))
            slf_attn_mask = broadcast_to(slf_attn_mask.astype(ms.int32)).astype(ms.bool_)
            # slf_attn_mask = np.broadcast_to(slf_attn_mask, (mask_dim0, max_len, mask_dim1))

            table = get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[: enc_seq.shape[1], :]
            table_dim0, table_dim1 = table.shape[0], table.shape[1]
            broadcast_to = ops.BroadcastTo((batch_size, table_dim0, table_dim1))
            dec_output = enc_seq + broadcast_to(self.expand_dims(table, 0))

        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            mask_dim0, mask_dim1 = mask.shape[0], mask.shape[1]
            slf_attn_mask = self.expand_dims(mask, 1)
            expand = ops.BroadcastTo((mask_dim0, max_len, mask_dim1))
            slf_attn_mask = expand(slf_attn_mask)

            position_enc = self.position_enc[:, :max_len, :]
            pos_dim1, pos_dim2 = position_enc.shape[1], position_enc.shape[2]
            expand = ops.BroadcastTo((batch_size, pos_dim1, pos_dim2))
            dec_output = enc_seq[:, :max_len, :] + expand(position_enc)

            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask
