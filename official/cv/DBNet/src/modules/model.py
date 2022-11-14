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

import mindspore.nn as nn
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
