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
"""RetinaNet_NAS_FPN."""

import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from src.retinahead import RetinaNetHead
from src.resnet import resnet50
from src.nasfpn import NASFPN
from src.loss import SigmoidFocalClassificationLoss


class retinanetNASFPN(nn.Cell):
    '''
        retinanet_nasfpn network

    Args:
        config (dict): retinanet_nasfpn config.
        is_training (bool): training or eval mode.

    Returns:
        Tensor, the prediction of the network.
    '''
    def __init__(self, config, is_training=True):
        super(retinanetNASFPN, self).__init__()
        self.backbone = resnet50()
        self.neck = NASFPN(in_channels=config.nasfpn_input_channels, out_channels=config.nasfpn_out_channel,
                           num_outs=config.nasfpn_num_outs, stack_times=config.nasfpn_stack_times,
                           start_level=config.nasfpn_start_level, end_level=config.nasfpn_end_level)
        self.head = RetinaNetHead(config)
        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, x):
        """Forward function."""
        _, _, C3, C4, C5 = self.backbone(x)
        features = self.neck((C3, C4, C5))
        pred_loc, pred_label = self.head(features)
        if not self.is_training:
            pred_label = self.activation(pred_label)
        return pred_loc, pred_label

class retinanetWithLossCell(nn.Cell):
    """"
    Provide retinanet training loss through network.

    Args:
        network (Cell): The training network.
        config (dict): retinanet_nasfpn config.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, network, config):
        super(retinanetWithLossCell, self).__init__()
        self.network = network
        self.less = P.Less()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.expand_dims = P.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(config.gamma, config.alpha)
        self.loc_loss = nn.SmoothL1Loss()
        self.cast = P.Cast()

        self.network.to_float(mstype.float32)

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        """Forward function."""
        pred_loc, pred_label = self.network(x)
        pred_loc = self.cast(pred_loc, mstype.float32)
        pred_label = self.cast(pred_label, mstype.float32)

        mask = F.cast(self.less(0, gt_label), mstype.float32)
        num_matched_boxes = self.reduce_sum(F.cast(num_matched_boxes, mstype.float32))

        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_mean(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))

        return self.reduce_sum((loss_cls + loss_loc) /num_matched_boxes)

class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of retinanet_nasfpn network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
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

    def construct(self, *args):
        """Forward function."""
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss

class retinanetInferWithDecoder(nn.Cell):
    """
    retinanet_nasfpn Infer wrapper to decode the bbox locations.

    Args:
        network (Cell): the origin retinanet_nasfpn infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator
        config (dict): retinanet config
    Returns:
        Tensor, the locations for bbox after decoder representing (y0,x0,y1,x1)
        Tensor, the prediction labels.

    """
    def __init__(self, network, default_boxes, config):
        super(retinanetInferWithDecoder, self).__init__()
        self.network = network
        self.default_boxes = default_boxes
        self.prior_scaling_xy = config.prior_scaling[0]
        self.prior_scaling_wh = config.prior_scaling[1]

    def construct(self, x):
        """Forward function."""
        pred_loc, pred_label = self.network(x)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_bbox_wh + default_bbox_xy
        pred_wh = P.Exp()(pred_loc[..., 2:] * self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = P.Concat(-1)((pred_xy_0, pred_xy_1))
        pred_xy = P.Maximum()(pred_xy, 0)
        pred_xy = P.Minimum()(pred_xy, 1)
        return pred_xy, pred_label
