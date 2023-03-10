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
"""transformer"""
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as mnp
from mindspore import ops
from mindspore.common.initializer import initializer

from src.init_weights import KaimingUniform
from src.init_weights import UniformBias


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return ops.ReLU()
    if activation == "gelu":
        return ops.GeLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def linear(input_arr, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out_features, in_features)`
        - Bias: :math:`(out_features)`
        - Output: :math:`(N, *, out_features)`
    """
    if input_arr.ndim == 2 and bias is not None:
        # fused op is marginally faster
        ret = ops.BatchMatMul()(input_arr, weight.T) + bias
    else:
        output = mnp.matmul(input_arr, weight.T)
        if bias is not None:
            output += bias
        ret = output
    return ret


def with_pos_embed(tensor, pos):
    """with pos embed"""
    return tensor if pos is None else tensor + pos


class Transformer(nn.Cell):
    """transformer"""
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        encoder_layers = nn.CellList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_encoder_layers)
        ])
        encoder_norm = nn.LayerNorm([d_model], epsilon=1e-5) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, encoder_norm)

        decoder_layers = nn.CellList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_decoder_layers)
        ])
        decoder_norm = nn.LayerNorm([d_model], epsilon=1e-5)
        self.decoder = TransformerDecoder(decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.d_model = d_model
        self.nhead = nhead

    def construct(self, src, mask, query_embed, pos_embed):
        """construct"""
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.view(bs, c, h * w).transpose(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, h * w).transpose(2, 0, 1)
        query_embed = ops.Tile()(ops.ExpandDims()(query_embed, 1), (1, bs, 1))
        mask = mask.view(bs, h * w)

        tgt = ops.ZerosLike()(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(0, 2, 1, 3)


class TransformerEncoder(nn.Cell):
    """transformer encoder"""
    def __init__(self, encoder_layers, norm=None):
        super().__init__()
        self.layers = encoder_layers
        self.norm = norm

    def construct(self, src, src_key_padding_mask=None, pos=None):
        """construct"""
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Cell):
    """transformer decoder"""
    def __init__(self, decoder_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = decoder_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def construct(self, tgt, memory,
                  tgt_key_padding_mask=None, memory_key_padding_mask=None,
                  pos=None, query_pos=None):
        """construct"""
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            output = ops.Stack()(intermediate)
            return output
        output = ops.ExpandDims()(output, 0)
        return output


class TransformerEncoderLayer(nn.Cell):
    """transformer encoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward,
                                weight_init=KaimingUniform(),
                                bias_init=UniformBias([dim_feedforward, d_model]))
        self.linear2 = nn.Dense(dim_feedforward, d_model,
                                weight_init=KaimingUniform(),
                                bias_init=UniformBias([d_model, dim_feedforward]))

        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.activation = _get_activation_fn(activation)
        self.drop0 = nn.Dropout(p=dropout)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

    def construct(self, src, src_key_padding_mask=None, pos=None):
        """construct"""
        q = k = with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
        src = src + self.drop0(src2)
        src = self.norm1(src)

        src2 = self.linear1(src)
        src2 = self.activation(src2)
        src2 = self.drop1(src2)
        src2 = self.linear2(src2)

        src = src + self.drop2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Cell):
    """transformer decoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward,
                                weight_init=KaimingUniform(),
                                bias_init=UniformBias([dim_feedforward, d_model]))
        self.linear2 = nn.Dense(dim_feedforward, d_model,
                                weight_init=KaimingUniform(),
                                bias_init=UniformBias([d_model, dim_feedforward]))

        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm3 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.drop0 = nn.Dropout(p=dropout)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)
        self.activation = _get_activation_fn(activation)

    def construct(self, tgt, memory,
                  tgt_key_padding_mask=None, memory_key_padding_mask=None,
                  pos=None, query_pos=None):
        """construct"""
        q = k = with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.drop0(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=with_pos_embed(tgt, query_pos),
                                   key=with_pos_embed(memory, pos),
                                   value=memory,
                                   key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.drop1(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.drop2(self.activation(self.linear1(tgt))))

        tgt = tgt + self.drop3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MultiheadAttention(nn.Cell):
    """multi head attention"""
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.q_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mstype.float32))
        self.k_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mstype.float32))
        self.v_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mstype.float32))

        self.q_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mstype.float32))
        self.k_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mstype.float32))
        self.v_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mstype.float32))

        self.out_proj = nn.Dense(embed_dim, embed_dim, weight_init=KaimingUniform())
        self.drop = nn.Dropout(p=dropout)

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  key_padding_mask: Tensor,
                  need_weights: bool = True):
        """construct"""
        tgt_len, bsz, embed_dim = query.shape
        scaling = self.head_dim ** -0.5

        q = linear(query, self.q_in_proj_weight, self.q_in_proj_bias)
        k = linear(key, self.k_in_proj_weight, self.k_in_proj_bias)
        v = linear(value, self.v_in_proj_weight, self.v_in_proj_bias)

        q = q * scaling

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        src_len = k.shape[1]

        attn_output_weights = ops.BatchMatMul()(q, k.transpose(0, 2, 1))

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = ops.Tile()(
                ops.ExpandDims()(ops.ExpandDims()(key_padding_mask, 1), 2),
                (1, self.num_heads, tgt_len, 1)
            )
            attn_output_weights = attn_output_weights - key_padding_mask * 10000.
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = ops.Softmax(axis=-1)(attn_output_weights)
        attn_output_weights = self.drop(attn_output_weights)

        attn_output = ops.BatchMatMul()(attn_output_weights, v)
        attn_output = attn_output.transpose(1, 0, 2).view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        return attn_output
