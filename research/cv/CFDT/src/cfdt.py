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


import copy
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
from mindspore.common.initializer import initializer, XavierUniform
from src.model_utils.misc import inverse_sigmoid
from src.backbone import ptnt_ti_patch16
from src.deformable_transformer import build_deformable_transformer


def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


class Detector(nn.Cell):
    """ This is a combination of "Swin with RAM" and a "Neck-free Deformable Decoder" """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, with_box_refine=False):
        """ Initializes the model.
        Parameters:
            backbone: mindspore module of the backbone to be used. See backbone.py
            transformer: mindspore module of the transformer architecture. See deformable_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries (i.e., det tokens). This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Dense(
            in_channels=hidden_dim, out_channels=num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.backbone = backbone
        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        # [PATCH] token channel reduction for the input to transformer decoder
        num_backbone_outs = len(backbone.num_channels)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            if _ >= 0:
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.SequentialCell([
                    #  1x1 conv
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, pad_mode='pad',
                              has_bias=True, weight_init='xavier_uniform', bias_init='zeros'),
                    nn.GroupNorm(32, hidden_dim),
                ]))
        input_proj_list.append(nn.SequentialCell([
            # 1x1 conv
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, pad_mode='pad',
                      has_bias=True, weight_init='xavier_uniform', bias_init='zeros'),
            nn.GroupNorm(32, hidden_dim),
        ]))
        self.input_proj = nn.CellList(input_proj_list)
        # initialize the projection layer for [PATCH] tokens
        for proj in self.input_proj:
            proj[0].weight = initializer(
                XavierUniform(), proj[0].weight.shape, mindspore.float32)
        self.fusion = None
        self.tgt_proj = nn.SequentialCell([
            # 1x1 conv
            nn.Conv2d(in_channels=backbone.num_channels[-2], out_channels=hidden_dim, kernel_size=1,
                      pad_mode='pad', has_bias=True, weight_init='xavier_uniform', bias_init='zeros'),
            nn.GroupNorm(32, hidden_dim),
        ])
        self.query_pos_proj = nn.SequentialCell([
            # 1x1 conv
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, pad_mode='pad', has_bias=True,
                      weight_init='xavier_uniform', bias_init='zeros'),
            nn.GroupNorm(32, hidden_dim),
        ])
        inner_input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = backbone.inner_dims[_]
            if _ >= 0:
                inner_input_proj_list.append(nn.SequentialCell([
                    #  1x1 conv
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim // 16, kernel_size=1, pad_mode='pad',
                              has_bias=True, weight_init='xavier_uniform', bias_init='zeros'),
                    nn.GroupNorm(32 // 16, hidden_dim // 16),
                ]))
        inner_input_proj_list.append(nn.SequentialCell([
            # 1x1 conv
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim // 16, kernel_size=1, pad_mode='pad',
                      has_bias=True, weight_init='xavier_uniform', bias_init='zeros'),
            nn.GroupNorm(32 // 16, hidden_dim // 16),
        ]))

        self.inner_input_proj_list = nn.CellList(inner_input_proj_list)
        num_pred = transformer.decoder.num_layers + 1

        # set up all required nn.Cell for additional techniques
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.class_embed = nn.CellList(
                [self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.CellList(
                [self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

    def construct(self, *inputs):
        """ The forward step of ViDT

        Parameters:
            The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dictionary having the key and value pairs below:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
                            If iou_aware is True, "pred_ious" is also returns as one of the key in "aux_outputs"
        """
        x, mask = inputs
        features, det_tgt, det_pos, inner_features = self.backbone(x,
                                                                   mask)  # 将图像输入到backbone中，获得outer和inner，以及det tokens

        det_tgt = self.tgt_proj(ops.expand_dims(
            det_tgt, -1)).squeeze(-1).transpose(0, 2, 1)
        det_pos = self.query_pos_proj(ops.expand_dims(
            det_pos, -1)).squeeze(-1).transpose(0, 2, 1)

        srcs = []
        srcs_inner = []
        if self.fusion is None:
            for l, src in enumerate(features):
                srcs.append(self.input_proj[l](src))

            for l, src_inner in enumerate(inner_features):
                srcs_inner.append(self.inner_input_proj_list[l](src_inner))

        masks = []
        for l, src in enumerate(srcs):
            # resize mask
            _mask = ops.Cast()(
                ops.interpolate(ops.Cast()(mask[None], mindspore.float32), sizes=(src.shape[-2], src.shape[-1]),
                                mode="bilinear", coordinate_transformation_mode='half_pixel')[0], mindspore.bool_)
            masks.append(_mask)
            assert mask is not None

        masks_inner = []
        for l, src in enumerate(srcs_inner):
            _mask = ops.Cast()(
                ops.interpolate(ops.Cast()(mask[None], mindspore.float32), sizes=(src.shape[-2], src.shape[-1]),
                                mode="bilinear", coordinate_transformation_mode='half_pixel')[0], mindspore.bool_)
            masks_inner.append(_mask)
            assert masks_inner is not None

        outputs_classes = []
        outputs_coords = []

        # neck
        hs, init_reference, inter_references, _ = self.transformer(
            srcs, masks, srcs_inner, masks_inner, det_tgt, det_pos)
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.class_embed[lvl](hs[lvl])
            # bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = ops.Sigmoid()(tmp)

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # stack all predictions made from each decoding layers
        outputs_class = ops.stack(outputs_classes)
        outputs_coord = ops.stack(outputs_coords)

        if self.aux_loss and self.transformer.decoder.num_layers > 0:
            out = ops.Concat(axis=-1)([outputs_class, outputs_coord])
        else:
            out = ops.Concat(axis=-1)([outputs_class[-1], outputs_coord[-1]])
        return out


def _set_aux_loss(self, outputs_class, outputs_coord):
    return [{'pred_logits': a.asnumpy(), 'pred_boxes': b.asnumpy()}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Cell):
    """ simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([])
        for n, k in zip([input_dim] + h, h + [output_dim]):
            self.layers.append(nn.Dense(
                in_channels=n, out_channels=k, weight_init='XavierUniform', has_bias=True))

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = P.ReLU()(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_cfdt(config, is_teacher=False):
    if config.dataset_file == 'coco':
        num_classes = config.num_classes
    if config.backbone_name == 'ptnt_t':
        backbone = ptnt_ti_patch16(pretrained=config.pre_trained)
    else:
        raise ValueError(f'backbone {config.backbone_name} not supported')

    backbone.finetune_det(method=None, det_token_num=config.det_token_num,
                          pos_dim=config.reduced_dim)

    deform_transformers = build_deformable_transformer(config)
    model = Detector(
        backbone,
        deform_transformers,
        num_classes=num_classes,
        num_queries=config.det_token_num,
        aux_loss=config.aux_loss,
        with_box_refine=config.with_box_refine,
    )

    return model
