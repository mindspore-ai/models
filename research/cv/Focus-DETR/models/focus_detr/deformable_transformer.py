# Copyright 2023 Huawei Technologies Co., Ltd
#
import copy
import random
from typing import Optional
import numpy as np

import mindspore as ms
import mindspore.numpy as ms_np
import mindspore.ops as ops
from mindspore import Parameter, Tensor, nn

from .ms_deform_attn import MSDeformAttn
from .utils import MLP, _get_activation_fn, gen_encoder_output_proposals, gen_sineembed_for_position, inverse_sigmoid


def save_ms_tensor(x):
    x = x.asnumpy()
    np.save("test_ms.npy", x)


class MultiheadAttention(nn.Cell):
    def __init__(self, d_model, n_head, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj = nn.Dense(self.embed_dim, 3 * self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.split = ops.Split(-1, 3)
        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.scaling = self.head_dim**-0.5

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


class MultiheadCrossAttention(nn.Cell):
    def __init__(self, d_model, n_head, dropout=0.0):
        super(MultiheadCrossAttention, self).__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = self.embed_dim // self.num_heads

        self.dense1 = nn.Dense(self.embed_dim, self.embed_dim)
        self.dense2 = nn.Dense(self.embed_dim, self.embed_dim)
        self.dense3 = nn.Dense(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.split = ops.Split(-1, 3)

        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.scaling = self.head_dim**-0.5

    def construct(self, query, key, value):
        tgt_len, bsz, embed_dim = query.shape

        q = self.dense1(query)
        k = self.dense2(key)
        v = self.dense3(value)

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


class DeformableTransformer(nn.Cell):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,
                 query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type="roi_align",
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type="ca",
                 module_seq=("sa", "ca", "ffn"),
                 # for dn
                 embed_init_tgt=False,
                 use_detached_boxes_dec_out=False,
    ):
        super().__init__()
        self.rho = 0.5
        self.enc_mask_predictor = MaskPredictor(d_model, d_model)
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        assert query_dim == 4
        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder
        assert layer_share_type in [None, "encoder", "decoder", "both"]
        enc_layer_share = layer_share_type in ["encoder", "both"]
        dec_layer_share = layer_share_type in ["decoder", "both"]

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]

        if deformable_encoder:
            encoder_layer = EncoderDeformableDecoderLayer(
                d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
            )
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm([d_model], epsilon=1e-5) if normalize_before else None

        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
            d_model=d_model,
            num_queries=num_queries,
            deformable_encoder=deformable_encoder,
            enc_layer_share=enc_layer_share,
            two_stage_type=two_stage_type,
        )

        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(
                d_model,
                dim_feedforward,
                dropout,
                activation,
                num_feature_levels,
                nhead,
                dec_n_points,
                use_deformable_box_attn=use_deformable_box_attn,
                box_attn_type=box_attn_type,
                key_aware_type=key_aware_type,
                decoder_sa_type=decoder_sa_type,
                module_seq=module_seq,
            )

        else:
            raise NotImplementedError
        decoder_norm = nn.LayerNorm([d_model], epsilon=1e-5)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            modulate_hw_attn=modulate_hw_attn,
            num_feature_levels=num_feature_levels,
            deformable_decoder=deformable_decoder,
            decoder_query_perturber=decoder_query_perturber,
            dec_layer_number=dec_layer_number,
            rm_dec_query_scale=rm_dec_query_scale,
            dec_layer_share=dec_layer_share,
            use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = Parameter(
                    Tensor(np.zeros((num_feature_levels, d_model)), ms.float32),
                    name="level_embed",
                    requires_grad=True,
                )
            else:
                self.level_embed = None
        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type == "standard":
            self.enc_output = nn.Dense(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm([d_model], epsilon=1e-5)
            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = Parameter(
                    Tensor(np.zeros((two_stage_pat_embed, d_model)), ms.float32),
                    name="pat_embed_for_2stage",
                    requires_grad=True,
                )

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)
            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None
        ##
        if two_stage_type == "no":
            self.init_ref_points(num_queries)
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != "no" or num_patterns == 0:
                assert (
                    dec_layer_number[0] == num_queries
                ), f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert (
                    dec_layer_number[0] == num_queries * num_patterns
                ), f"dec_layer_number[0]({dec_layer_number[0]}) != " \
                   f"num_queries({num_queries}) * num_patterns({num_patterns})"

        self.rm_self_attn_layers = rm_self_attn_layers
        self.alpha = Parameter(Tensor(np.zeros(3), ms.float32), name="alpha", requires_grad=True)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = ms_np.sum(~mask[:, :, 0], axis=1)
        valid_W = ms_np.sum(~mask[:, 0, :], axis=1)
        valid_ratio_h = valid_H.astype("float32") / H
        valid_ratio_w = valid_W.astype("float32") / W
        valid_ratio = ms_np.stack([valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes, process_output=True):
        r"""
        Input:
            - memory: bs, \sum{hw}, d_model
            - memory_padding_mask: bs, \sum{hw}
            - spatial_shapes: nlevel, 2
            - learnedwh: 2
        Output:
            - output_memory: bs, \sum{hw}, d_model
            - output_proposals: bs, \sum{hw}, 4
        """
        N_, = memory.shape[0]
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            H_ = int(H_.asnumpy())
            W_ = int(W_.asnumpy())
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)

            mask_flatten_ = mask_flatten_.astype("bool")
            not_mask = ~mask_flatten_
            valid_H = not_mask[:, :, 0, 0].sum(axis=1)
            valid_W = not_mask[:, 0, :, 0].sum(axis=1)

            start = Tensor(0, ms.float32)
            end = Tensor(H_ - 1, ms.float32)

            h = H_
            H_m = ops.linspace(start, end, h)

            start = Tensor(0, ms.float32)
            end = Tensor(W_ - 1, ms.float32)
            w = W_
            W_m = ops.linspace(start, end, w)
            grid_y, grid_x = ops.meshgrid((H_m, W_m), indexing="ij")

            expand_dims = ops.ExpandDims()
            grid = ops.concat([expand_dims(grid_x, -1), expand_dims(grid_y, -1)], -1)

            scale = ops.concat([expand_dims(valid_W, -1), expand_dims(valid_H, -1)], 1).view(N_, 1, 1, 2)
            grid = expand_dims(grid, 0)
            grid = (ops.broadcast_to(grid, (N_, -1, -1, -1)) + 0.5) / scale

            wh = ops.ones_like(grid) * 0.05 * (2.0**lvl)

            proposal = ops.concat([grid, wh], -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_

        output_proposals = ops.concat(proposals, 1)
        output_proposals_1 = output_proposals > 0.01
        output_proposals_2 = output_proposals < 0.99
        output_proposals_valid = ops.logical_and(output_proposals_1, output_proposals_2)
        output_proposals_valid = output_proposals_valid.all(-1, keep_dims=True)
        output_proposals = ops.log(output_proposals / (1 - output_proposals))  # unsigmoid
        output_proposals = ops.masked_fill(output_proposals, expand_dims(memory_padding_mask, -1), float("inf"))
        output_proposals = ops.masked_fill(output_proposals, ~output_proposals_valid, float("inf"))
        output_memory = memory
        output_memory = ops.masked_fill(output_memory, expand_dims(memory_padding_mask, -1), float(0))
        output_memory = ops.masked_fill(output_memory, ~output_proposals_valid, float(0))

        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals, (~memory_padding_mask).sum(axis=-1)

    def upsamplelike(self, inputs):
        src, target = inputs
        return ops.interpolate(
            src, sizes=(target.shape[2], target.shape[3]), mode="bilinear", coordinate_transformation_mode="half_pixel"
        )

    def construct(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.reshape((bs, c, h * w))
            src = ops.transpose(src, (0, 2, 1))

            mask = mask.reshape((bs, h * w))

            pos_embed = pos_embed.reshape((bs, c, h * w))  # bs, hw, c
            pos_embed = ops.transpose(pos_embed, (0, 2, 1))
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].reshape(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            #########################################
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = ops.concat(src_flatten, 1)
        mask_flatten = ops.concat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = ops.concat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c

        spatial_shapes = ms.Tensor(spatial_shapes)

        spatial_shapes = spatial_shapes.astype("Float32")
        spatial_shapes_data = spatial_shapes.prod(1).cumsum(0)[:-1]
        spatial_shapes_data = spatial_shapes_data.astype("Int64")

        zeros_data = ms.Tensor(np.array([0]))

        level_start_index = ops.concat([zeros_data, spatial_shapes_data])
        valid_ratios = ms_np.stack([self.get_valid_ratio(m) for m in masks], axis=1)
        # Begin token pruning
        if self.rho or self.use_enc_aux_loss:
            expand_dims = ops.ExpandDims()
            backbone_output_memory, _, valid_token_nums = self.gen_encoder_output_proposals(
                src_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes, process_output=bool(self.rho)
            )

            sparse_token_nums = (valid_token_nums * self.rho).asnumpy().astype(int) + 1
            backbone_topk = int(max(sparse_token_nums))
            self.sparse_token_nums = sparse_token_nums
            backbone_topk = min(backbone_topk, backbone_output_memory.shape[1])

            backbone_topk1 = int(backbone_topk * 0.8)
            backbone_topk2 = int(backbone_topk * 0.6)
            backbone_topk3 = int(backbone_topk * 0.6)
            backbone_topk4 = int(backbone_topk * 0.4)
            backbone_topk5 = int(backbone_topk * 0.2)

            backbone_output_memory_4 = backbone_output_memory[:, level_start_index[4] :, :]
            backbone_mask_prediction_4 = self.enc_mask_predictor(backbone_output_memory_4)

            backbone_output_memory_3 = backbone_output_memory[
                :, level_start_index[3] : level_start_index[4], :
            ]  # [B, H3*W3, C]=[2,950,256]

            temp = expand_dims(backbone_mask_prediction_4 * backbone_output_memory_4, 1)
            backbone_output_memory_3 = (
                backbone_output_memory_3
                + ops.squeeze(self.upsamplelike((temp, expand_dims(backbone_output_memory_3, 1))), 1) * self.alpha[0]
            )
            backbone_mask_prediction_3 = self.enc_mask_predictor(backbone_output_memory_3)

            backbone_output_memory_2 = backbone_output_memory[:, level_start_index[2] : level_start_index[3], :]

            temp = expand_dims(backbone_mask_prediction_3 * backbone_output_memory_3, 1)
            backbone_output_memory_2 = (
                backbone_output_memory_2
                + ops.squeeze(self.upsamplelike((temp, expand_dims(backbone_output_memory_2, 1))), 1) * self.alpha[0]
            )
            backbone_mask_prediction_2 = self.enc_mask_predictor(backbone_output_memory_2)

            backbone_output_memory_1 = backbone_output_memory[:, level_start_index[1] : level_start_index[2], :]
            temp = expand_dims(backbone_mask_prediction_2 * backbone_output_memory_2, 1)

            backbone_output_memory_1 = (
                backbone_output_memory_1
                + ops.squeeze(self.upsamplelike((temp, expand_dims(backbone_output_memory_1, 1))), 1) * self.alpha[1]
            )
            backbone_mask_prediction_1 = self.enc_mask_predictor(backbone_output_memory_1)

            backbone_output_memory_0 = backbone_output_memory[:, level_start_index[0] : level_start_index[1], :]
            temp = expand_dims(backbone_mask_prediction_1 * backbone_output_memory_1, 1)
            backbone_output_memory_0 = (
                backbone_output_memory_0
                + ops.squeeze(self.upsamplelike((temp, expand_dims(backbone_output_memory_0, 1))), 1) * self.alpha[2]
            )
            backbone_mask_prediction_0 = self.enc_mask_predictor(backbone_output_memory_0)

            backbone_mask_prediction = ops.concat(
                (
                    backbone_mask_prediction_0,
                    backbone_mask_prediction_1,
                    backbone_mask_prediction_2,
                    backbone_mask_prediction_3,
                    backbone_mask_prediction_4,
                ),
                axis=1,
            )
            backbone_mask_prediction = ops.squeeze(backbone_mask_prediction, -1)

            backbone_mask_prediction = backbone_mask_prediction.masked_fill(
                mask_flatten, backbone_mask_prediction.min()
            )
            backbone_topk_proposals = []

            backbone_topk_proposals0 = ops.TopK(sorted=True)(backbone_mask_prediction, backbone_topk)[1]  # 存在一定误差#
            backbone_topk_proposals1 = ops.TopK(sorted=True)(backbone_mask_prediction, backbone_topk1)[1]
            backbone_topk_proposals2 = ops.TopK(sorted=True)(backbone_mask_prediction, backbone_topk2)[1]
            backbone_topk_proposals3 = ops.TopK(sorted=True)(backbone_mask_prediction, backbone_topk3)[1]
            backbone_topk_proposals4 = ops.TopK(sorted=True)(backbone_mask_prediction, backbone_topk4)[1]
            backbone_topk_proposals5 = ops.TopK(sorted=True)(backbone_mask_prediction, backbone_topk5)[1]

            backbone_topk_proposals.append(backbone_topk_proposals0)
            backbone_topk_proposals.append(backbone_topk_proposals1)
            backbone_topk_proposals.append(backbone_topk_proposals2)
            backbone_topk_proposals.append(backbone_topk_proposals3)
            backbone_topk_proposals.append(backbone_topk_proposals4)
            backbone_topk_proposals.append(backbone_topk_proposals5)

        enc_topk_proposals = enc_refpoint_embed = None

        memory, _, _ = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            # spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            topk_inds=backbone_topk_proposals,
            sparse_token_nums=ms.Tensor(sparse_token_nums),
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )
        if self.two_stage_type == "standard":
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, input_hw
            )

            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            if self.two_stage_pat_embed > 0:
                pass
            if self.two_stage_add_query_num > 0:
                pass
            if self.enc_out_class_embed is None:
                raise ValueError("enc_out_class_embed not defined")
            # pylint: disable=not-callable
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            if self.enc_out_bbox_embed is None:
                raise ValueError("enc_out_bbox_embed not defined")
            # pylint: disable=not-callable
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid

            topk = self.num_queries
            enc_outputs_class_unselected_max = enc_outputs_class_unselected.max(-1)

            topk_proposals = ops.TopK(sorted=True)(enc_outputs_class_unselected_max, topk)[1]

            topk_proposals_2 = topk_proposals.reshape(-1)
            refpoint_embed_undetach = ops.gather(enc_outputs_coord_unselected, topk_proposals_2, 1)

            refpoint_embed_ = refpoint_embed_undetach
            sigmoid = ops.Sigmoid()
            init_box_proposal = sigmoid(ops.gather(output_proposals, topk_proposals_2, 1))  # sigmoid
            tgt_undetach = ops.gather(output_memory, topk_proposals_2, 1)

            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.embedding_table[:, None, :]

                tgt_ = ops.transpose(tgt_, (1, 0, 2))
            else:
                tgt_ = tgt_undetach.detach()
            if refpoint_embed is not None:
                pass
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_
        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))

        hs, references = self.decoder(
            tgt=ops.transpose(tgt, (1, 0, 2)),
            memory=ops.transpose(memory, (1, 0, 2)),
            memory_key_padding_mask=mask_flatten,
            pos=ops.transpose(lvl_pos_embed_flatten, (1, 0, 2)),
            refpoints_unsigmoid=ops.transpose(refpoint_embed, (1, 0, 2)),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
        )
        if self.two_stage_type == "standard":
            if self.two_stage_keep_all_tokens:
                pass
            else:
                expand_dims = ops.ExpandDims()
                hs_enc = expand_dims(tgt_undetach, 0)
                sigmoid = ops.Sigmoid()
                ref_enc = expand_dims(sigmoid(refpoint_embed_undetach), 0)
        else:
            hs_enc = ref_enc = None
        return hs, references, hs_enc, ref_enc, init_box_proposal


class DeformableTransformerEncoderLayer(nn.Cell):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        add_channel_attention=False,
        use_deformable_box_attn=False,
        box_attn_type="roi_align",
    ):
        super().__init__()
        # self attention
        if use_deformable_box_attn:
            self.self_attn = MSDeformableBoxAttention(
                d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type
            )
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        self.dropout1 = nn.Dropout(keep_prob=1 - dropout)

        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)
        # ffn
        self.linear1 = nn.Dense(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(keep_prob=1 - dropout)
        self.linear2 = nn.Dense(d_ffn, d_model)
        self.dropout3 = nn.Dropout(keep_prob=1 - dropout)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn("dyrelu", d_model=d_model)
            self.norm_channel = nn.LayerNorm([d_model], epsilon=1e-5)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def construct(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))
        return src


#
class TransformerEncoder(nn.Cell):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        d_model=256,
        num_queries=300,
        deformable_encoder=False,
        enc_layer_share=False,
        enc_layer_dropout_prob=None,
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
    ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0
        self.two_stage_type = two_stage_type
        if two_stage_type in ["enceachlayer", "enclayer1"]:
            _proj_layer = nn.Dense(d_model, d_model)
            _norm_layer = nn.LayerNorm([d_model], epsilon=1e-5)
            if two_stage_type == "enclayer1":
                self.enc_norm = [_norm_layer]
                self.enc_proj = [_proj_layer]
            else:
                self.enc_norm = [copy.deepcopy(_norm_layer) for i in range(num_layers - 1)]
                self.enc_proj = [copy.deepcopy(_proj_layer) for i in range(num_layers - 1)]

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            start = Tensor(0.5, ms.float32)
            end = Tensor(H_ - 0.5, ms.float32)
            h = int(H_.asnumpy())
            H_m = ops.linspace(start, end, h)
            start = Tensor(0.5, ms.float32)
            end = Tensor(W_ - 0.5, ms.float32)
            w = int(W_.asnumpy())
            W_m = ops.linspace(start, end, w)
            ref_y, ref_x = ops.meshgrid((H_m, W_m), indexing="ij")
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = ms_np.stack([ref_x, ref_y], axis=-1)
            reference_points_list.append(ref)
        reference_points = ops.concat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def construct(
        self,
        src: Tensor,
        pos: Tensor,
        spatial_shapes,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        topk_inds: Tensor,
        sparse_token_nums: Tensor,
        ref_token_index: Optional[Tensor] = None,
        ref_token_coord: Optional[Tensor] = None,
    ):
        """
        Input:
            - focus_detr: [bs, sum(hi*wi), 256]
            - pos: pos embed for focus_detr. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outputs:
            - output: [bs, sum(hi*wi), 256]
        """
        if self.two_stage_type in ["no", "standard", "enceachlayer", "enclayer1"]:
            assert ref_token_index is None
        output = src
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios)
        intermediate_output = []
        intermediate_ref = []
        expand_dims = ops.ExpandDims()
        if ref_token_index is not None:
            input_index = ms.numpy.tile(expand_dims(ref_token_index, -1), [1, 1, self.d_model])
            out_i = ops.gather(output, input_index, axis=1)
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        B_, N_, S_, P_ = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = pos
        for layer_id, layer in enumerate(self.layers):
            tgt_index = ms.numpy.tile(expand_dims(topk_inds[layer_id], -1), [1, 1, output.shape[-1]])
            tgt = ops.gather_elements(output, dim=1, index=tgt_index)
            pos_index = ms.numpy.tile(expand_dims(topk_inds[layer_id], -1), [1, 1, output.shape[-1]])
            pos = ops.gather_elements(ori_pos, dim=1, index=pos_index)
            res_index = ms.numpy.tile(expand_dims(topk_inds[layer_id], -1), [1, 1, S_ * P_])
            reference_points = ops.gather_elements(ori_reference_points.view(B_, N_, -1), index=res_index, dim=1).view(
                B_, -1, S_, P_
            )
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                if self.deformable_encoder:
                    tgt = layer(
                        src=output,
                        pos=pos,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask,
                        tgt=tgt,
                    )
                else:
                    output = None
            if sparse_token_nums is None:
                output = ops.tensor_scatter_elements(
                    output, ms.numpy.tile(ops.expand_dims(topk_inds[layer_id], -1), [1, 1, tgt.size(-1)]), tgt
                )
            else:
                outputs = []

                for i in range(topk_inds[layer_id].shape[0]):
                    outputs.append(
                        ops.tensor_scatter_elements(
                 output[i],
                 ms.numpy.tile(expand_dims(topk_inds[layer_id][i], -1), [1, tgt.shape[-1]]),
                 tgt[i][: sparse_token_nums[i]],
                 axis=0,
                        )
                    )
                output = ops.stack(outputs, axis=0)
        if self.norm is not None:
            output = self.norm(output)
        intermediate_output = intermediate_ref = None
        return output, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Cell):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        modulate_hw_attn=False,
        num_feature_levels=1,
        deformable_decoder=False,
        decoder_query_perturber=None,
        dec_layer_number=None,
        rm_dec_query_scale=False,
        dec_layer_share=False,
        dec_layer_dropout_prob=None,
        use_detached_boxes_dec_out=False,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError

        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def construct(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        refpoints_unsigmoid=None,
        # for memory
        level_start_index=None,  # num_levels
        spatial_shapes=None,  # bs, num_levels, 2
        valid_ratios=None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        intermediate = []
        sigmoid = ops.Sigmoid()
        reference_points = sigmoid(refpoints_unsigmoid)
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:  #
                pass
            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = (
                        reference_points[:, :, None] * ops.Concat(axis=-1)([valid_ratios, valid_ratios])[None, :]
                    )  # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed_input = reference_points_input[:, :, 0, :]
                query_sine_embed = gen_sineembed_for_position(query_sine_embed_input)  # nq, bs, 256*2
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points)  # nq, bs, 256*2
                reference_points_input = None
            # pylint: disable=not-callable
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            dropflag = False

            if not dropflag:
                output = layer(
                    tgt=output,  # N,B,C=900,1,256
                    tgt_query_pos=query_pos,  # N,B,C=900,1,256
                    tgt_query_sine_embed=query_sine_embed,  # N,B,2C=900,1,512
                    tgt_key_padding_mask=tgt_key_padding_mask,  # None
                    tgt_reference_points=reference_points_input,  # N,B,NP,4=900,1,5,4
                    memory=memory,  # hw,B,C
                    memory_key_padding_mask=memory_key_padding_mask,  # [1, hw]
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_pos=pos,  # hw,B,C
                    self_attn_mask=tgt_mask,  # None
                    cross_attn_mask=memory_mask,  # None
                )

            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)

                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid

                sigmoid = ops.Sigmoid()
                new_reference_points = sigmoid(outputs_unsig)
                reference_points = new_reference_points
                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                    pass
                ref_points.append(reference_points)

            tmp_out = self.norm(output)
            intermediate.append(tmp_out)

            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                pass
        return [
            [ops.transpose(itm_out, (1, 0, 2)) for itm_out in intermediate],
            [ops.transpose(itm_refpoint, (1, 0, 2)) for itm_refpoint in ref_points],
        ]


class DeformableTransformerDecoderLayer(nn.Cell):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_deformable_box_attn=False,
        box_attn_type="roi_align",
        key_aware_type=None,
        decoder_sa_type="ca",
        module_seq=("sa", "ca", "ffn"),
    ):
        super().__init__()
        self.module_seq = module_seq
        if not use_deformable_box_attn:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(keep_prob=1 - dropout)
        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.self_attn = MultiheadCrossAttention(d_model=d_model, n_head=n_heads)  # MultiheadCrossAttention
        self.dropout2 = nn.Dropout(keep_prob=1 - dropout)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.linear1 = nn.Dense(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(keep_prob=1 - dropout)
        self.linear2 = nn.Dense(d_ffn, d_model)
        self.dropout4 = nn.Dropout(keep_prob=1 - dropout)
        self.norm3 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]
        if decoder_sa_type == "ca_content":
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        if self.self_attn is not None:
            if self.decoder_sa_type == "sa":
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)

            elif self.decoder_sa_type in ["ca_label", "ca_content"]:
                pass
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))
        return tgt

    def forward_ca(
        self,
        # for tgt
        tgt,  # nq, bs, d_model
        tgt_query_pos=None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed=None,  # pos for query. Sine(pos)
        tgt_key_padding_mask=None,
        tgt_reference_points=None,  # nq, bs, 4
        # for memory
        memory=None,  # hw, bs, d_model
        memory_key_padding_mask=None,
        memory_level_start_index=None,  # num_levels
        memory_spatial_shapes=None,  # bs, num_levels, 2
        memory_pos=None,  # pos for memory
        # sa
        self_attn_mask=None,  # mask used for self-attention
        cross_attn_mask=None,  # mask used for cross-attention
    ):
        input1 = ops.transpose(self.with_pos_embed(tgt, tgt_query_pos), (1, 0, 2))
        input2 = ops.transpose(tgt_reference_points, (1, 0, 2, 3))
        input3 = ops.transpose(memory, (1, 0, 2))

        tgt2 = self.cross_attn(
            input1, input2, input3, memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask
        )
        tgt3 = ops.transpose(tgt2, (1, 0, 2))
        tgt = tgt + self.dropout1(tgt3)
        tgt = self.norm1(tgt)
        return tgt

    def construct(
        self,
        # for tgt
        tgt,  # nq, bs, d_model
        tgt_query_pos=None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed=None,  # pos for query. Sine(pos)
        tgt_key_padding_mask=None,
        tgt_reference_points=None,  # nq, bs, 4
        # for memory
        memory=None,  # hw, bs, d_model
        memory_key_padding_mask=None,
        memory_level_start_index=None,  # num_levels
        memory_spatial_shapes=None,  # bs, num_levels, 2
        memory_pos=None,  # pos for memory
        # sa
        self_attn_mask=None,  # mask used for self-attention
        cross_attn_mask=None,  # mask used for cross-attention
    ):
        for funcname in self.module_seq:
            if funcname == "ffn":
                tgt = self.forward_ffn(tgt)
            elif funcname == "ca":
                tgt = self.forward_ca(
                    tgt,
                    tgt_query_pos,
                    tgt_query_sine_embed,
                    tgt_key_padding_mask,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_pos,
                    self_attn_mask,
                    cross_attn_mask,
                )
            elif funcname == "sa":
                tgt = self.forward_sa(
                    tgt,
                    tgt_query_pos,
                    tgt_query_sine_embed,
                    tgt_key_padding_mask,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_pos,
                    self_attn_mask,
                    cross_attn_mask,
                )
            else:
                raise ValueError("unknown funcname {}".format(funcname))
        return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return [module for i in range(N)]
    return nn.SequentialCell([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(args):
    decoder_query_perturber = None
    use_detached_boxes_dec_out = False

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,
        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,
        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,
        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,
        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,
        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,
    )


class EncoderDeformableDecoderLayer(nn.Cell):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(keep_prob=1 - dropout)
        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)

        # self attention
        self.self_attn = MultiheadCrossAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(keep_prob=1 - dropout)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        # ffn
        self.linear1 = nn.Dense(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(keep_prob=1 - dropout)
        self.linear2 = nn.Dense(d_ffn, d_model)
        self.dropout4 = nn.Dropout(keep_prob=1 - dropout)
        self.norm3 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.linear_class = nn.Dense(d_model, 90)
        self.topk_sa = 400

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def construct(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None, tgt=None):
        ori_tgt = tgt
        score_tgt = self.linear_class(tgt)
        ms_topk = ops.TopK()
        expand_dims = ops.ExpandDims()
        select_tgt_index = ms_topk(score_tgt.max(-1)[0], self.topk_sa)[1]  # bs, nq

        select_tgt = ops.GatherD()(tgt, 1, ms_np.tile(expand_dims(select_tgt_index, -1), [1, 1, 256]))
        select_pos = ops.GatherD()(pos, 1, ms_np.tile(expand_dims(select_tgt_index, -1), [1, 1, 256]))

        q = k = self.with_pos_embed(select_tgt, select_pos)
        tgt2 = self.self_attn(q.transpose(1, 0, 2), k.transpose(1, 0, 2), select_tgt.transpose(1, 0, 2)).transpose(
            1, 0, 2
        )
        tgt = select_tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = ops.tensor_scatter_elements(
            ori_tgt, ms_np.tile(ops.expand_dims(select_tgt_index, -1), [1, 1, tgt.shape[-1]]), tgt, axis=1
        )

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)

        return tgt


class MaskPredictor(nn.Cell):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.SequentialCell(
            nn.LayerNorm([in_dim], epsilon=1e-5), nn.Dense(in_dim, h_dim), nn.GELU(approximate=False)
        )
        self.layer2 = nn.SequentialCell(
            nn.Dense(h_dim, h_dim // 2),
            nn.GELU(approximate=False),
            nn.Dense(h_dim // 2, h_dim // 4),
            nn.GELU(approximate=False),
            nn.Dense(h_dim // 4, 1),
        )

    def construct(self, x):
        z = self.layer1(x.view(x.shape))
        new_tensor = ops.split(z, output_num=2, axis=-1)
        z_local, z_global = new_tensor[0], new_tensor[1]
        z_global = ops.broadcast_to(z_global.mean(axis=1, keep_dims=True), (-1, z_local.shape[1], -1))
        z = ops.concat((z_local, z_global), axis=-1)
        out = self.layer2(z.view(z.shape))
        return out
