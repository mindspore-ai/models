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
"""TrainOneStepWithClipGNAndEMA."""
import logging as logger

import mindspore.nn as nn
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.trainer.ema import EMACell

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


class TrainOneStepWithClipGNAndEMA(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStepWithEMA"""

    def __init__(self, network, optimizer,
                 use_global_norm=False, clip_global_norm_value=1.0,
                 scale_sense=1.0, with_ema=False, ema_decay=0.9999):
        super(TrainOneStepWithClipGNAndEMA, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.with_ema = with_ema
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        if self.with_ema:
            self.ema_model = EMACell(self.weights, ema_decay=ema_decay)

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_global_norm:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
            loss = F.depend(loss, self.optimizer(grads))
            if self.with_ema:
                self.ema_model(self.weights)
        else:
            self.print("=============Over Flow, skipping=============")
        return loss


def create_train_one_step(args, net_with_loss, optimizer, log=logger):
    """get_train_one_step cell"""
    if args.use_dynamic_loss_scale:
        log.info(f"=> Using DynamicLossScaleUpdateCell")
        scale_manager = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(
            loss_scale_value=2 ** 24, scale_factor=2, scale_window=2000
        )
    else:
        log.info(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_manager = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)
    if args.use_ema and not args.use_global_norm:
        net_with_loss = TrainOneStepWithClipGNAndEMA(
            net_with_loss, optimizer, scale_sense=scale_manager,
            with_ema=args.use_ema, ema_decay=args.ema_decay)
    elif args.use_ema and args.use_global_norm:
        net_with_loss = TrainOneStepWithClipGNAndEMA(
            net_with_loss, optimizer, use_global_norm=args.use_global_norm,
            clip_global_norm_value=args.clip_gn_value, scale_sense=scale_manager,
            with_ema=args.use_ema, ema_decay=args.ema_decay)
    else:
        net_with_loss = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scale_manager)
    return net_with_loss
