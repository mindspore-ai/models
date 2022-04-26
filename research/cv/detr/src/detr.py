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
"""detr"""

import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.common import initializer as init
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context

from src.backbone import Backbone
from src.backbone import resnet50
from src.init_weights import KaimingUniform
from src.init_weights import UniformBias
from src.position_encoding import PositionEmbeddingSine
from src.transformer import Transformer


class DeTR(nn.Cell):
    """Detection Transformer"""
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Dense(hidden_dim, num_classes + 1,
                                    weight_init=KaimingUniform(),
                                    bias_init=UniformBias([num_classes + 1, hidden_dim]))
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim,
                                        embedding_table=init.Normal(sigma=1.0))
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim,
                                    kernel_size=1, has_bias=True,
                                    weight_init=KaimingUniform(),
                                    bias_init=UniformBias([hidden_dim, backbone.num_channels]))
        self.backbone = backbone
        self.aux_loss = aux_loss

    def construct(self, *inputs):
        """construct"""
        images, masks = inputs

        feature, mask_interp, pos = self.backbone(images, masks)
        hs = self.transformer(self.input_proj(feature),
                              mask_interp,
                              self.query_embed.embedding_table,
                              pos)
        outputs_class = self.class_embed(hs)
        outputs_coord = ops.Sigmoid()(self.bbox_embed(hs))

        if self.aux_loss:
            output = ops.Concat(axis=-1)([outputs_class, outputs_coord])
        else:
            output = ops.Concat(axis=-1)([outputs_class[-1], outputs_coord[-1]])
        return output


class MLP(nn.Cell):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([
            nn.Dense(n, k, weight_init=KaimingUniform(), bias_init=UniformBias([k, n]))
            for n, k in zip([input_dim] + h, h + [output_dim])
        ])

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
        self.scale_sense = Parameter(Tensor(initial_scale_sense, dtype=mstype.float32), name="scale_sense")
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
        self.scale_sense.set_data(Tensor(data, dtype=mstype.float32))

    def construct(self, *inputs):
        """construct"""
        pred = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs, self.scale_sense * 1.)
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
        ce_w = self.criterion.weight_dict['loss_ce']
        bbox_w = self.criterion.weight_dict['loss_bbox']
        giou_w = self.criterion.weight_dict['loss_giou']
        sens_np = np.concatenate(
            [ce_w * loss_np['loss_ce_grad_src'],
             bbox_w * loss_np['loss_bbox_grad_src'] + giou_w * loss_np['loss_giou_grad_src']],
            axis=-1)

        if self.aux_loss:
            sens_np_aux = np.stack([
                np.concatenate([
                    ce_w * loss_np[f'loss_ce_grad_src_{i}'],
                    bbox_w * loss_np[f'loss_bbox_grad_src_{i}'] +
                    giou_w * loss_np[f'loss_giou_grad_src_{i}']
                ], axis=-1)
                for i in range(self.n_dec - 1)
            ])
            sens_np = np.concatenate([sens_np_aux, np.expand_dims(sens_np, 0)])
        return sens_np

    def __call__(self, inputs, gt):
        """call"""
        # first pass data through the network for calculating the loss and its gradients
        network_output = self.one_step_cell.network(*inputs)
        if self.aux_loss:
            n_aux_outs = network_output.shape[0] - 1
            outputs = {
                'pred_logits': network_output[-1, :, :, :92].asnumpy(),
                'pred_boxes': network_output[-1, :, :, 92:].asnumpy(),
                'aux_outputs': [
                    {'pred_logits': network_output[i, ..., :92].asnumpy(),
                     'pred_boxes': network_output[i, ..., 92:].asnumpy()}
                    for i in range(n_aux_outs)
                ]
            }
        else:
            outputs = {'pred_logits': network_output[:, :, :92].asnumpy(),
                       'pred_boxes': network_output[:, :, 92:].asnumpy()}

        gt = {k: v.asnumpy() for k, v in gt.items()}

        loss_np = self.criterion(outputs, gt)
        loss_value = sum(loss_np[k] * self.criterion.weight_dict[k]
                         for k in loss_np.keys() if k in self.criterion.weight_dict)[0]

        # update sensitivity parameter
        sens_np = self._get_sens_np(loss_np)

        self.one_step_cell.set_sense(Tensor(sens_np))
        # second pass data through the network for backpropagation step
        pred = self.one_step_cell(*inputs)
        return loss_value, pred, network_output


def build_detr(cfg):
    """build detr"""
    position_encodding = PositionEmbeddingSine(cfg.hidden_dim // 2, normalize=True)
    backbone = Backbone(resnet50(cfg.backbone_pretrain), position_encodding)
    transformer = Transformer(d_model=cfg.hidden_dim,
                              dropout=cfg.dropout,
                              nhead=cfg.nheads,
                              dim_feedforward=cfg.dim_feedforward,
                              num_encoder_layers=cfg.enc_layers,
                              num_decoder_layers=cfg.dec_layers,
                              normalize_before=cfg.pre_norm,
                              return_intermediate_dec=True)

    num_classes = cfg.num_classes

    detr = DeTR(backbone,
                transformer,
                num_classes,
                cfg.num_queries,
                cfg.aux_loss)
    return detr
