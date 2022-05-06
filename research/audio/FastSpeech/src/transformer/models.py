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
"""Model script."""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops

from src.cfg.config import config as hp
from src.transformer import constants
from src.transformer.layers import FFTBlock


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    Sinusoid position encoding table.
    """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return Tensor(sinusoid_table, dtype=mstype.float32)


class Encoder(nn.Cell):
    """Encoder."""
    def __init__(
            self,
            n_src_vocab,
            len_max_seq,
            d_word_vec,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_model,
            d_inner,
            dropout,
    ):
        super().__init__()

        n_position = len_max_seq + 1
        pretrained_embs = get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0)

        self.src_word_emb = nn.Embedding(
            n_src_vocab,
            d_word_vec,
            padding_idx=constants.PAD,
        )

        self.position_enc = nn.Embedding(
            n_position,
            d_word_vec,
            embedding_table=pretrained_embs,
            padding_idx=0,
        )

        self.layer_stack = nn.CellList(
            [
                FFTBlock(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)
            ]
        )

        self.equal = ops.Equal()
        self.not_equal = ops.NotEqual()
        self.expand_dims = ops.ExpandDims()
        self.pad = constants.PAD
        self.broadcast = ops.BroadcastTo((-1, hp.character_max_length, -1))

    def construct(self, src_seq, src_pos):
        """
        Create mask and forward to FFT blocks.

        Args:
            src_seq (Tensor): Tokenized text sequence. Shape (hp.batch_size, hp.character_max_length).
            src_pos (Tensor): Positions of the sequences. Shape (hp.batch_size, hp.character_max_length).

        Returns:
            enc_output (Tensor): Encoder output.
        """
        # Prepare masks
        padding_mask = self.equal(src_seq, self.pad)
        slf_attn_mask = self.broadcast(self.expand_dims(padding_mask.astype(mstype.float32), 1))
        slf_attn_mask_bool = slf_attn_mask.astype(mstype.bool_)

        non_pad_mask_bool = self.expand_dims(self.not_equal(src_seq, self.pad), -1)
        non_pad_mask = non_pad_mask_bool.astype(mstype.float32)

        # Forward
        enc_output = self.src_word_emb(src_seq.astype('int32')) + self.position_enc(src_pos.astype('int32'))

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask_bool,
            )

        return enc_output


class Decoder(nn.Cell):
    """Decoder."""
    def __init__(
            self,
            len_max_seq,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_model,
            d_inner,
            dropout
    ):

        super().__init__()

        n_position = len_max_seq + 1

        pretrained_embs = get_sinusoid_encoding_table(n_position, d_model, padding_idx=0)

        self.position_enc = nn.Embedding(
            n_position,
            d_model,
            embedding_table=pretrained_embs,
            padding_idx=0,
        )

        self.layer_stack = nn.CellList(
            [
                FFTBlock(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)
            ]
        )

        self.pad = constants.PAD
        self.equal = ops.Equal()
        self.not_equal = ops.NotEqual()
        self.expand_dims = ops.ExpandDims()
        self.broadcast = ops.BroadcastTo((-1, hp.mel_max_length, -1))

    def construct(self, enc_seq, enc_pos):
        """
        Create mask and forward to FFT blocks.
        """
        # Prepare masks
        padding_mask = self.equal(enc_pos, self.pad)
        slf_attn_mask = self.broadcast(self.expand_dims(padding_mask.astype(mstype.float32), 1))
        slf_attn_mask_bool = slf_attn_mask.astype(mstype.bool_)

        non_pad_mask_bool = self.expand_dims(self.not_equal(enc_pos, self.pad), -1)
        non_pad_mask = non_pad_mask_bool.astype(mstype.float32)

        # Forward
        dec_output = enc_seq + self.position_enc(enc_pos.astype(mstype.int32))

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask_bool)

        return dec_output
