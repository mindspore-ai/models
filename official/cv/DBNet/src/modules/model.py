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
# This file refers to the project https://github.com/MhLiao/DB.git
"""DBNet models."""

from mindspore import ops, nn
from .backbone import get_backbone
from .detector import SegDetector


def get_dbnet(net, config, isTrain=True):
    if net == "DBnet":
        return DBnet(config, isTrain)
    if net == "DBnetPP":
        return DBnetPP(config, isTrain)
    raise ValueError(f"Not support net {net}, net should be in [DBnet, DBnetPP]")


class DBnet(nn.Cell):
    def __init__(self, config, isTrain=True):
        super(DBnet, self).__init__(auto_prefix=False)

        self.backbone = get_backbone(config.backbone.initializer)(config.backbone.pretrained,
                                                                  config.backbone.backbone_ckpt)
        seg = config.segdetector
        self.segdetector = SegDetector(in_channels=seg.in_channels, inner_channels=seg.inner_channels,
                                       k=seg.k, bias=seg.bias, adaptive=seg.adaptive,
                                       serial=seg.serial, training=isTrain)

    def construct(self, img):
        pred = self.backbone(img)
        pred = self.segdetector(pred)

        return pred


class DBnetPP(nn.Cell):
    def __init__(self, config, isTrain=True):
        super(DBnetPP, self).__init__(auto_prefix=False)

        self.backbone = get_backbone(config.backbone.initializer)(config.backbone.pretrained,
                                                                  config.backbone.backbone_ckpt)
        seg = config.segdetector
        self.segdetector = SegDetector(in_channels=seg.in_channels, inner_channels=seg.inner_channels,
                                       k=seg.k, bias=seg.bias, adaptive=seg.adaptive,
                                       serial=seg.serial, training=isTrain, concat_attention=True,
                                       attention_type=seg.attention_type)

    def construct(self, img):
        pred = self.backbone(img)
        pred = self.segdetector(pred)
        return pred


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)

        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img, gt, gt_mask, thresh_map, thresh_mask):
        pred = self._backbone(img)
        loss = self._loss_fn(pred, gt, gt_mask, thresh_map, thresh_mask)

        return loss

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


_grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))


class TrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    r"""Network training with loss scaling"""
    def __init__(self, network, optimizer, scale_sense, clip_grad=False, force_update=False):
        super(TrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.clip_grad = clip_grad
        self.force_update = force_update

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * scaling_sens.astype(loss.dtype)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        if self.clip_grad:
            grads = ops.clip_by_global_norm(grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if ops.logical_or(self.force_update, not overflow):
            loss = ops.depend(loss, self.optimizer(grads))

        return loss, cond, scaling_sens
