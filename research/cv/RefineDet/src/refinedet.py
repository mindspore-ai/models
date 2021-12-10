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
"""RefineDet network structure"""

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from .vgg16_for_refinedet import vgg16
from .resnet101_for_refinedet import resnet
from .multibox import MultiBox
from .l2norm import L2Norm

def _make_conv_layer(channels, use_bn=False, use_relu=True, kernel_size=3, padding=0):
    """make convolution layer for refinedet"""
    in_channels = channels[0]
    layers = []
    for out_channels in channels[1:]:
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, pad_mode="pad", padding=padding))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            layers.append(nn.ReLU())
        in_channels = out_channels
    return layers

def _make_deconv_layer(channels, use_bn=False, use_relu=True, kernel_size=3, padding=0, stride=1):
    """make deconvolution layer for TCB"""
    in_channels = channels[0]
    layers = []
    for out_channels in channels[1:]:
        layers.append(nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, pad_mode="pad", padding=padding, stride=stride))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.SequentialCell(layers)

class TCB(nn.Cell):
    """TCB block for transport features from ARM to ODM"""
    def __init__(self, arm_source_num, in_channels, normalization, use_bn=False):
        super(TCB, self).__init__()
        self.layers = []
        self.t_num = arm_source_num
        self.add = P.Add()
        for idx in range(self.t_num):
            self.layers.append([])
            if normalization:
                if normalization[idx] != -1:
                    self.layers[idx] += [L2Norm(in_channels[idx], normalization[idx])]

            self.layers[idx] += _make_conv_layer([in_channels[idx], 256], use_bn=use_bn, padding=1)
            if idx + 1 == self.t_num:
                self.layers[idx] += [nn.SequentialCell(_make_conv_layer([256, 256, 256], use_bn=use_bn, padding=1))]
            else:
                self.layers[idx] += _make_conv_layer([256, 256], use_bn=use_bn, use_relu=False, padding=1)
                self.layers[idx] += [nn.SequentialCell(_make_conv_layer([256, 256, 256], use_bn=use_bn, padding=1))]
        self.tcb0 = nn.SequentialCell(self.layers[0][:-1])
        self.deconv0 = _make_deconv_layer([256, 256], use_bn=use_bn, kernel_size=2, stride=2)
        self.p0 = self.layers[0][-1]
        self.tcb1 = nn.SequentialCell(self.layers[1][:-1])
        self.deconv1 = _make_deconv_layer([256, 256], use_bn=use_bn, kernel_size=2, stride=2)
        self.p1 = self.layers[1][-1]
        self.tcb2 = nn.SequentialCell(self.layers[2][:-1])
        self.deconv2 = _make_deconv_layer([256, 256], use_bn=use_bn, kernel_size=2, stride=2)
        self.p2 = self.layers[2][-1]
        self.tcb3 = nn.SequentialCell(self.layers[3][:-1])
        self.p3 = self.layers[3][-1]

    def construct(self, x):
        """construct network"""
        outputs = ()
        tmp = x[3]
        tmp = self.tcb3(tmp)
        tmp = self.p3(tmp)
        outputs += (tmp,)
        tmp = x[2]
        tmp = self.tcb2(tmp)
        tmp = self.add(tmp, self.deconv2(outputs[0]))
        tmp = self.p2(tmp)
        outputs = (tmp,) + outputs
        tmp = x[1]
        tmp = self.tcb1(tmp)
        tmp = self.add(tmp, self.deconv1(outputs[0]))
        tmp = self.p1(tmp)
        outputs = (tmp,) + outputs
        tmp = x[0]
        tmp = self.tcb0(tmp)
        tmp = self.add(tmp, self.deconv0(outputs[0]))
        tmp = self.p0(tmp)
        outputs = (tmp,) + outputs
        return outputs

class ARM(nn.Cell):
    """anchor refined module"""
    def __init__(self, backbone, config, is_training=True):
        super(ARM, self).__init__()
        self.layer = []
        self.layers = {}
        self.backbone = backbone
        self.multi_box = MultiBox(config, 2, config.extra_arm_channels)
        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, x):
        """construct network"""
        outputs = self.backbone(x)
        multi_feature = outputs
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.is_training:
            pred_label = self.activation(pred_label)
        pred_loc = F.cast(pred_loc, ms.float32)
        pred_label = F.cast(pred_label, ms.float32)
        return outputs, pred_loc, pred_label

class ODM(nn.Cell):
    """object detecion module"""
    def __init__(self, config, is_training=True):
        super(ODM, self).__init__()
        self.layer = []
        self.layers = {}
        self.multi_box = MultiBox(config, config.num_classes, config.extra_odm_channels)
        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, x):
        """construct network"""
        outputs = x
        multi_feature = outputs
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.is_training:
            pred_label = self.activation(pred_label)
        pred_loc = F.cast(pred_loc, ms.float32)
        pred_label = F.cast(pred_label, ms.float32)
        return pred_loc, pred_label

class RefineDet(nn.Cell):
    """refinedet network"""
    def __init__(self, backbone, config, is_training=True):
        super(RefineDet, self).__init__()
        self.backbone = backbone
        self.is_training = is_training
        self.arm = ARM(backbone, config, is_training)
        self.odm = ODM(config, is_training)
        self.tcb = TCB(len(config.arm_source), config.extra_arm_channels, config.L2normalizations)

    def construct(self, x):
        """construct network"""
        arm_out, arm_pre_loc, arm_pre_label = self.arm(x)
        tcb_out = self.tcb(arm_out)
        odm_pre_loc, odm_pre_label = self.odm(tcb_out)
        return arm_pre_loc, arm_pre_label, odm_pre_loc, odm_pre_label, arm_out

def refinedet_vgg16(config, is_training=True):
    """return refinedet with vgg16"""
    return RefineDet(backbone=vgg16(), config=config, is_training=is_training)


def refinedet_resnet101(config, is_training=True):
    """return refinedet with resnet101"""
    return RefineDet(backbone=resnet(), config=config, is_training=is_training)

class RefineDetInferWithDecoder(nn.Cell):
    """
    RefineDet Infer wrapper to decode the bbox locations. (As detection layers in other forms)
    Args:
        network (Cell): the origin ssd infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator
        config (dict): network config
    Returns:
        Tensor, the locations for bbox after decoder representing (y0,x0,y1,x1)
        Tensor, the prediction labels.
    """
    def __init__(self, network, default_boxes, config):
        super(RefineDetInferWithDecoder, self).__init__()
        self.network = network
        self.default_boxes = default_boxes
        self.prior_scaling_xy = config.prior_scaling[0]
        self.prior_scaling_wh = config.prior_scaling[1]
        self.objectness_thre = config.objectness_thre
        self.softmax1 = nn.Softmax()
        self.softmax2 = nn.Softmax()

    def construct(self, x):
        """construct network"""
        _, arm_label, odm_loc, odm_label, _ = self.network(x)

        arm_label = self.softmax1(arm_label)
        pred_loc = odm_loc
        pred_label = self.softmax2(odm_label)
        pred_label = odm_label
        arm_object_conf = arm_label[:, :, 1:]
        no_object_index = F.cast(arm_object_conf > self.objectness_thre, ms.float32)
        pred_label = pred_label * no_object_index.expand_as(pred_label)

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
