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
# ------------------------------------------------------------------------
# This file was modified modified for CFDT from ViDT/NAVER https://github.com/naver-ai/vidt


"""Transforms and data augmentation for both image and bbox."""

import copy
import mindspore
import mindspore.ops.operations as P
import mindspore.numpy as np
from mindspore.ops import stop_gradient
from mindspore import nn
from mindspore import ops
from src.model_utils.misc import inverse_sigmoid, DropPath1D
from src.ms_deform_attn import MSDeformAttn


class DeformableTransformer(nn.Cell):
    """ A Deformable Transformer for the neck in a detector

    The transformer encoder is completely removed for ViDT
    Parameters:
        d_model: the channel dimension for attention [default=256]
        nhead: the number of heads [default=8]
        num_decoder_layers: the number of decoding layers [default=6]
        dim_feedforward: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        return_intermediate_dec: whether to return all the indermediate outputs [default=True]
        num_feature_levels: the number of scales for extracted features [default=4]
        dec_n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
    """

    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=True, num_feature_levels=4, dec_n_points=4,
                 drop_path=0.):
        super().__init__()

        self.d_model = d_model
        self.num_feature_levels = num_feature_levels
        self.nhead = nhead
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points,
                                                          drop_path=drop_path)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.reference_points = nn.Dense(in_channels=d_model, out_channels=2)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = ops.ReduceSum()(ops.Cast()(~mask[:, :, 0], mindspore.float32), 1)
        valid_W = ops.ReduceSum()(ops.Cast()(~mask[:, 0, :], mindspore.float32), 1)
        valid_ratio_h = valid_H / H
        valid_ratio_w = valid_W / W
        valid_ratio = ops.Stack(axis=-1)([valid_ratio_w, valid_ratio_h])
        return valid_ratio

    def construct(self, srcs, masks, srcs_inner, masks_inner, tgt, query_pos):
        """ The forward step of the decoder

        Parameters:
            srcs: [Patch] tokens
            masks: input padding mask
            tgt: [DET] tokens
            query_pos: [DET] token pos encodings

        Returns:
            hs: calibrated [DET] tokens
            init_reference_out: init reference points
            inter_references_out: intermediate reference points for box refinement
            enc_token_class_unflat: info. for token labeling
        """

        # prepare input for the Transformer decoder
        src_flatten = []
        mask_flatten = []
        spatial_shapes_list = []

        src_flatten_inner = []
        mask_flatten_inner = []
        spatial_shapes_inner_list = []

        for _, (src_out, mask) in enumerate(zip(srcs, masks)):
            bs, _, h, w = src_out.shape
            spatial_shapes_list.append((h, w))
            src_out = ops.reshape(src_out, (src_out.shape[0], src_out.shape[1], -1)).transpose(0, 2, 1)
            mask = ops.reshape(mask, (mask.shape[0], -1))
            src_flatten.append(src_out)
            mask_flatten.append(mask)

        # wei 1103
        for _, (src_in, mask) in enumerate(zip(srcs_inner, masks_inner)):
            _, _, h2, w2 = src_in.shape
            # spatial_shape_inner = (h, w)
            spatial_shapes_inner_list.append((h2, w2))
            src_in = ops.reshape(src_in, (src_in.shape[0], src_in.shape[1], -1)).transpose(0, 2, 1)
            mask = ops.reshape(mask, (mask.shape[0], -1))
            src_flatten_inner.append(src_in)
            mask_flatten_inner.append(mask)

        src_flatten = P.Concat(1)(src_flatten)
        mask_flatten = P.Concat(1)(mask_flatten)

        spatial_shapes = np.array(spatial_shapes_list, np.float32)
        level_start_index = ops.concat(
            (ops.Zeros()((1,), mindspore.float32), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = ops.Stack(axis=1)([self.get_valid_ratio(m) for m in masks])

        # inner
        src_flatten_inner = P.Concat(1)(src_flatten_inner)
        mask_flatten_inner = P.Concat(1)(mask_flatten_inner)
        spatial_shapes_inner = np.array(spatial_shapes_inner_list, np.float32)
        level_start_index_inner = P.Concat()(
            (ops.Zeros()((1,), mindspore.float32), spatial_shapes_inner.prod(1).cumsum(0)[:-1]))
        memory = src_flatten  # [b,xxxx,256]

        memory_inner = src_flatten_inner

        bs, _, _ = memory.shape
        tgt = tgt  # [DET] tokens [b,100,256]
        query_pos = ops.function.broadcast_to(query_pos,
                                              (bs, query_pos.shape[1], -1))  # [DET] token pos encodings [b,100,256]
        enc_token_class_unflat = None
        reference_points = nn.Sigmoid()(self.reference_points(query_pos))  # [b,100,2]
        init_reference_out = reference_points  # query_pos -> reference point
        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory, memory_inner,
                                            spatial_shapes_list, spatial_shapes_inner_list, level_start_index,
                                            level_start_index_inner, valid_ratios, query_pos, mask_flatten,
                                            mask_flatten_inner)

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out, enc_token_class_unflat


class MultiheadAttention(nn.Cell):
    def __init__(self, d_model, n_head, dropout=0.):
        """

        :param d_model: width of tensor/embedding dim
        :param n_head: output of mutlithead attention/num_heads
        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj = nn.Dense(self.embed_dim, 3 * self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.split = ops.Split(-1, 3)
        self.expand_dims = P.ExpandDims()
        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.scaling = self.head_dim ** -0.5

    def construct(self, query, key, value):
        tgt_len, bsz, embed_dim = query.shape
        qkv = self.in_proj(query).view(tgt_len, bsz, 3, embed_dim).transpose((2, 0, 1, 3))
        qkv_v = self.in_proj(value).view(tgt_len, bsz, 3, embed_dim).transpose((2, 0, 1, 3))
        q = qkv[0:1]
        k = qkv[1:2]
        v = qkv_v[2:3]
        q = ops.Squeeze(0)(q)
        k = ops.Squeeze(0)(k)
        v = ops.Squeeze(0)(v)
        q = q * self.scaling
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        attn_output_weights = ops.matmul(q, k.transpose((0, 2, 1)))  # bs x (HW + 1) x (HW + 1)
        attn_output_weights = self.softmax(attn_output_weights)  # bs x (HW + 1) x (HW + 1)
        attn_output = ops.matmul(attn_output_weights, v)  # bs x (HW + 1) x h
        attn_output = self.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class DeformableTransformerDecoderLayer(nn.Cell):
    """ A decoder layer.

    Parameters:
        d_model: the channel dimension for attention [default=256]
        d_ffn: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        n_levels: the number of scales for extracted features [default=4]
        n_heads: the number of heads [default=8]
        n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, drop_path=0.):
        super().__init__()

        # [DET x PATCH] deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=(d_model,), epsilon=1e-05)

        # [DET x DET] self-attention
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=(d_model,), epsilon=1e-05)

        # ffn for multi-heaed
        self.linear1 = nn.Dense(in_channels=d_model, out_channels=d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear2 = nn.Dense(in_channels=d_ffn, out_channels=d_model)
        self.dropout4 = nn.Dropout(p=dropout)
        self.norm3 = nn.LayerNorm(normalized_shape=(d_model,), epsilon=1e-05)

        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3((self.activation()(self.linear1(tgt)))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def construct(self, tgt, query_pos, reference_points, src, src_inner, src_spatial_shapes, src_spatial_shapes_inner,
                  level_start_index, level_start_index_inner, src_padding_mask=None, src_padding_mask_inner=None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(1, 0, 2), k.transpose(1, 0, 2), tgt.transpose(1, 0, 2)).transpose(1, 0, 2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Multi-scale deformable cross-attention in Eq. (1) in the ViDT paper
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_inner, src_spatial_shapes, src_spatial_shapes_inner, level_start_index,
                               level_start_index_inner, src_padding_mask, src_padding_mask_inner)

        if self.drop_path is None:
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # ffn
            tgt = self.forward_ffn(tgt)
        else:
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.drop_path(self.dropout4(tgt2))
            tgt = self.norm3(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Cell):
    """ A Decoder consisting of multiple layers

    Parameters:
        decoder_layer: a deformable decoding layer
        num_layers: the number of layers
        return_intermediate: whether to return intermediate results
    """

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def construct(self, tgt, reference_points, src, src_inner, src_spatial_shapes, src_spatial_shapes_inner,
                  src_level_start_index, src_level_start_index_inner, src_valid_ratios,
                  query_pos=None, src_padding_mask=None, src_padding_mask_inner=None):
        """ The forward step of the Deformable Decoder

        Parameters:
            tgt: [DET] tokens [b,100,256]
            reference_points: reference points for deformable attention [b,100,2]
            src: the [PATCH] tokens fattened into a 1-d sequence [b,xxxx,256]
            src_spatial_shapes: the spatial shape of each multi-scale feature map
            src_level_start_index: the start index to refer different scale inputs
            src_valid_ratios: the ratio of multi-scale feature maps
            query_pos: the pos encoding for [DET] tokens [b,100,256]
            src_padding_mask: the input padding mask

        Returns:
            output: [DET] tokens calibrated (i.e., object embeddings)
            reference_points: A reference points

            If return_intermediate = True, output & reference_points are returned from all decoding layers
        """
        output = tgt
        intermediate = []
        intermediate_reference_points = []

        if self.bbox_embed is not None:
            tmp = self.bbox_embed[0](output)
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = ops.Sigmoid()(new_reference_points)
            else:
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = ops.Sigmoid()(new_reference_points)
            reference_points = stop_gradient(new_reference_points)
        #

        if self.return_intermediate:
            intermediate.append(output)  # tgt
            intermediate_reference_points.append(reference_points)
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * P.Concat(-1)([src_valid_ratios, src_valid_ratios])[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            # deformable operation
            output = layer(output, query_pos, reference_points_input, src, src_inner, src_spatial_shapes,
                           src_spatial_shapes_inner, src_level_start_index, src_level_start_index_inner,
                           src_padding_mask, src_padding_mask_inner)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid + 1](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = ops.Sigmoid()(new_reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = ops.Sigmoid()(new_reference_points)
                reference_points = stop_gradient(new_reference_points)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return ops.Stack()(intermediate), ops.Stack()(intermediate_reference_points)
        return output, reference_points


def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""

    if activation == "relu":
        return ops.ReLU
    if activation == "gelu":
        return ops.Gelu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deformable_transformer(config):
    return DeformableTransformer(
        d_model=config.reduced_dim,
        nhead=config.nheads,
        num_decoder_layers=config.dec_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=config.num_feature_levels,
        dec_n_points=config.dec_n_points)
