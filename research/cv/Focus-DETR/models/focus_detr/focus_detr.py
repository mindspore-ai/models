# Copyright 2023 Huawei Technologies Co., Ltd
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
import copy
import numpy as np
from mindspore import Parameter, Tensor, context, ops, nn
from mindspore.common import initializer as init
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context

from models.focus_detr.backbone import Backbone, resnet50
from models.focus_detr.init_weights import KaimingUniform, UniformBias
from models.focus_detr.position_encoding import PositionEmbeddingSine

from .deformable_transformer import build_deformable_transformer
from .utils import inverse_sigmoid


class DeTR(nn.Cell):
    """Detection Transformer"""

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Dense(
            hidden_dim,
            num_classes + 1,
            weight_init=KaimingUniform(),
            bias_init=UniformBias([num_classes + 1, hidden_dim]),
        )
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim, embedding_table=init.Normal(sigma=1.0))
        self.input_proj = nn.Conv2d(
            backbone.num_channels,
            hidden_dim,
            kernel_size=1,
            has_bias=True,
            weight_init=KaimingUniform(),
            bias_init=UniformBias([hidden_dim, backbone.num_channels]),
        )
        self.backbone = backbone
        self.aux_loss = aux_loss

    def construct(self, *inputs):
        """construct"""
        images, masks = inputs

        feature, mask_interp, pos = self.backbone(images, masks)
        hs = self.transformer(self.input_proj(feature), mask_interp, self.query_embed.embedding_table, pos)
        outputs_class = self.class_embed(hs)
        outputs_coord = ops.Sigmoid()(self.bbox_embed(hs))

        if self.aux_loss:
            output = ops.Concat(axis=-1)([outputs_class, outputs_coord])
        else:
            output = ops.Concat(axis=-1)([outputs_class[-1], outputs_coord[-1]])
        return output


class MLP(nn.Cell):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList(
            [
                nn.Dense(n, k, weight_init=KaimingUniform(), bias_init=UniformBias([k, n]))
                for n, k in zip([input_dim] + h, h + [output_dim])
            ]
        )

    def construct(self, x):
        """construct"""
        for i, layer in enumerate(self.layers):
            x = ops.ReLU()(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TrainOneStepCellWithSense(nn.Cell):
    """train one step cell with sense"""

    def __init__(self, network, optimizer, initial_scale_sense, max_grad_norm=0.1):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.scale_sense = Parameter(Tensor(initial_scale_sense, dtype=ms.float32), name="scale_sense")
        self.reducer_flag = False
        self.grad_reducer = None
        self.max_grad_norm = max_grad_norm
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def set_sense(self, data):
        """set sense"""
        self.scale_sense.set_data(Tensor(data, dtype=ms.float32))

    def construct(self, *inputs):
        """construct"""
        pred = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs, self.scale_sense * 1.0)
        grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        pred = ops.depend(pred, self.optimizer(grads))
        return pred


class TrainOneStepWrapper:
    """train one step wrapper"""

    def __init__(self, one_step_cell, criterion, aux_loss=True, n_dec=6):
        self.one_step_cell = one_step_cell
        self.criterion = criterion
        self.aux_loss = aux_loss
        self.n_dec = n_dec

    def _get_sens_np(self, loss_np):
        """get sens np"""
        ce_w = self.criterion.weight_dict["loss_ce"]
        bbox_w = self.criterion.weight_dict["loss_bbox"]
        giou_w = self.criterion.weight_dict["loss_giou"]
        sens_np = np.concatenate(
            [
                ce_w * loss_np["loss_ce_grad_src"],
                bbox_w * loss_np["loss_bbox_grad_src"] + giou_w * loss_np["loss_giou_grad_src"],
            ],
            axis=-1,
        )

        if self.aux_loss:
            sens_np_aux = np.stack(
                [
                    np.concatenate(
                        [
                 ce_w * loss_np[f"loss_ce_grad_src_{i}"],
                 bbox_w * loss_np[f"loss_bbox_grad_src_{i}"] + giou_w * loss_np[f"loss_giou_grad_src_{i}"],
                        ],
                        axis=-1,
                    )
                    for i in range(self.n_dec - 1)
                ]
            )
            sens_np = np.concatenate([sens_np_aux, np.expand_dims(sens_np, 0)])
        return sens_np

    def __call__(self, inputs, gt):
        """call"""
        # first pass data through the network for calculating the loss and its gradients
        network_output = self.one_step_cell.network(*inputs)
        if self.aux_loss:
            n_aux_outs = network_output.shape[0] - 1
            outputs = {
                "pred_logits": network_output[-1, :, :, :92].asnumpy(),
                "pred_boxes": network_output[-1, :, :, 92:].asnumpy(),
                "aux_outputs": [
                    {
                        "pred_logits": network_output[i, ..., :92].asnumpy(),
                        "pred_boxes": network_output[i, ..., 92:].asnumpy(),
                    }
                    for i in range(n_aux_outs)
                ],
            }
        else:
            outputs = {
                "pred_logits": network_output[:, :, :92].asnumpy(),
                "pred_boxes": network_output[:, :, 92:].asnumpy(),
            }

        gt = {k: v.asnumpy() for k, v in gt.items()}

        loss_np = self.criterion(outputs, gt)
        loss_value = sum(
            loss_np[k] * self.criterion.weight_dict[k] for k in loss_np.keys() if k in self.criterion.weight_dict
        )[0]

        # update sensitivity parameter
        sens_np = self._get_sens_np(loss_np)

        self.one_step_cell.set_sense(Tensor(sens_np))
        # second pass data through the network for backpropagation step
        pred = self.one_step_cell(*inputs)
        return loss_value, pred, network_output


class Focus_DETR(nn.Cell):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(self,
                 backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 iter_update=False,
                 query_dim=2,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 # two stage
                 two_stage_type="no",  # ['no', 'standard']
                 two_stage_add_query_num=0,
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 decoder_sa_type="sa",
                 num_patterns=0,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box separately
                     >0 : given fixed number
                     -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads

        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.SequentialCell(
                        nn.Conv2d(in_channels, hidden_dim, has_bias=True, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.SequentialCell(
                        nn.Conv2d(
                 in_channels, hidden_dim, kernel_size=3, has_bias=True, stride=2, padding=1, pad_mode="pad"
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.SequentialCell(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.iter_update = iter_update

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Dense(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]

        self.bbox_embed = nn.SequentialCell(box_embed_layerlist)
        self.class_embed = nn.SequentialCell(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed
        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num

        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(two_stage_type)

        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
            self.refpoint_embed = None

            if self.two_stage_add_query_num > 0:
                pass
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]

    def construct(self, samples=None, targets=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                 Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                 dictionaries containing the two above keys for each decoder layer.
        """
        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src = feat["data"]
            mask = feat["mask"]
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1]["data"])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples["mask"]
                mask = ops.ResizeNearestNeighbor(size=src.shape[-2:])(m[None].astype("float32"))[0]
                mask = mask.astype("bool")

                pos_l = self.backbone.position_embedding(mask).astype(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        input_query_bbox = input_query_label = attn_mask = None
        hs, references, _, _, _ = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask
        )
        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(references[:-1], self.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            sigmoid = ops.Sigmoid()
            layer_outputs_unsig = sigmoid(layer_outputs_unsig)

            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = ops.stack(outputs_coord_list)
        outputs_class = ops.stack(
            [layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)]
        )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        return out


def build_focus_detr(args):
    """build detr"""
    num_classes = args.num_classes
    assert args.pe_temperatureH == args.pe_temperatureW, "pe_temperatureH  and pe_temperaturew  should  be   same"
    temperature = args.pe_temperatureH
    position_encodding = PositionEmbeddingSine(
        temperature=temperature, num_pos_feats=args.hidden_dim // 2, normalize=True
    )
    backbone = Backbone(resnet50(), position_encodding)
    transformer = build_deformable_transformer(args)
    try:
        dn_labelbook_size = args.dn_labelbook_size
    except Exception:
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except Exception:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except Exception:
        dec_pred_bbox_embed_share = True

    model = Focus_DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
    )
    return model
