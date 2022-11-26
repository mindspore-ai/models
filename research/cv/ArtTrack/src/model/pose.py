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

import mindspore as ms
import mindspore.numpy as np
from mindspore import nn
from mindspore.ops import stop_gradient
from mindspore import load_checkpoint, load_param_into_net
from src.model.losses import HuberLossWithWeight, MSELossWithWeight, WeightLoss
from src.model.resnet import util


class PredictionLayer(nn.Cell):
    """
    prediction layer
    """

    def __init__(self, cfg, name, in_channels, out_channels):
        """
        Args:
            cfg: net config
            name: prediction name
            in_channels: in channels
            out_channels: out channels
        """
        super(PredictionLayer, self).__init__()
        self.cfg = cfg
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d_transpose = nn.Conv2dTranspose(in_channels=self.in_channels,
                                                   out_channels=self.out_channels, kernel_size=3, stride=2,
                                                   bias_init='normal', has_bias=True)

    def construct(self, x):
        return self.conv2d_transpose(x)


class PoseNet(nn.Cell):
    """
    pose net
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: net config
        """
        super(PoseNet, self).__init__()
        self.cfg = cfg
        if cfg.context.device_target == "Ascend":
            if cfg.model_arts.IS_MODEL_ARTS:
                pretrained = cfg.model_arts.CACHE_INPUT + 'crop/' +'resnet101.ckpt'
            else:
                pretrained = cfg.under_line.DATASET_ROOT + 'resnet101.ckpt'
                print(pretrained)
            param_dict = load_checkpoint(pretrained)
        self.resnet101 = util.resnet_101(3, output_stride=16, global_pool=False)
        if cfg.context.device_target == "Ascend":
            load_param_into_net(self.resnet101, param_dict)
        self.resnet101.set_train(False)
        self.resnet101.set_grad(requires_grad=False)
        self.mean = np.array(self.cfg.mean_pixel)
        self.mean = self.mean.reshape([1, 3, 1, 1])

        self.part_pred = PredictionLayer(self.cfg, 'part_pred', 2048, self.cfg.num_joints)
        if cfg.location_refinement:
            self.locref = PredictionLayer(cfg, 'locref_pred', 2048, cfg.num_joints * 2)
        if cfg.pairwise_predict:
            self.pairwise_pred = PredictionLayer(cfg, 'pairwise_pred', 2048, cfg.num_joints * (cfg.num_joints - 1) * 2)
        if cfg.intermediate_supervision:
            self.part_pred_interm = PredictionLayer(cfg, 'part_pred_interm', cfg.intermediate_supervision_input,
                                                    self.cfg.num_joints)

    def get_im_centered(self, inputs):
        return inputs - self.mean

    def construct(self, inputs):
        im_centered = self.get_im_centered(inputs)
        net, intermediate = self.resnet101(im_centered)
        net = stop_gradient(net)
        intermediate = stop_gradient(intermediate)
        return net, intermediate


class PoseNetTest(nn.Cell):
    """
    pose net eval
    """

    def __init__(self, net, cfg):
        """
        Args:
            net: pose net
            cfg: net config
        """
        super(PoseNetTest, self).__init__()
        self.net = net
        self.cfg = cfg
        self.location_refinement = cfg.location_refinement
        self.pairwise_predict = cfg.pairwise_predict
        self.sigmoid = nn.Sigmoid()

    def construct(self, *inputs):
        features, _ = self.net(inputs[0])
        out = self.net.part_pred(features)
        pairwise_pred = None
        locref = None
        if self.pairwise_predict:
            pairwise_pred = self.net.pairwise_pred(features)
        if self.location_refinement:
            locref = self.net.locref(features)

        return self.sigmoid(out), pairwise_pred, locref

class PoseNetTestExport(nn.Cell):
    """
    pose net for export
    """

    def __init__(self, net, cfg):
        """
        Args:
            net: pose net
            cfg: net config
        """
        super(PoseNetTestExport, self).__init__()
        self.net = net
        self.cfg = cfg
        self.location_refinement = cfg.location_refinement
        self.pairwise_predict = cfg.pairwise_predict
        self.sigmoid = nn.Sigmoid()

    def construct(self, *inputs):
        features, _ = self.net(inputs[0])
        out = self.net.part_pred(features)
        pairwise_pred = None
        locref = None
        if self.pairwise_predict:
            pairwise_pred = self.net.pairwise_pred(features)
        if self.location_refinement:
            locref = self.net.locref(features)

        if (not self.pairwise_predict) and (not self.location_refinement):
            return self.sigmoid(out)
        if not self.pairwise_predict:
            return self.sigmoid(out), locref
        if not self.location_refinement:
            return self.sigmoid(out), pairwise_pred
        return self.sigmoid(out), pairwise_pred, locref


class PoseNetBaseLoss(nn.Cell):
    """
    pose net base loss
    """

    def __init__(self, net, cfg):
        super(PoseNetBaseLoss, self).__init__()
        self.net = net
        self.cfg = cfg


class PoseNetTotalLoss(PoseNetBaseLoss):
    """
    pose net total loss
    """

    def __init__(self, net, cfg):
        """
        Args:
            net: pose net
            cfg: net config
        """
        super(PoseNetTotalLoss, self).__init__(net, cfg)
        self.part_score_weights = 1.0
        self.sce = ms.ops.SigmoidCrossEntropyWithLogits()
        self.weight_loss = WeightLoss()
        self.pairwise_loss_func = HuberLossWithWeight() if self.cfg.pairwise_huber_loss else MSELossWithWeight()
        self.locref_loss_func = HuberLossWithWeight() if self.cfg.locref_huber_loss else MSELossWithWeight()
        self.location_refinement = cfg.location_refinement
        self.pairwise_predict = cfg.pairwise_predict
        self.intermediate_supervision = cfg.intermediate_supervision
        self.locref_loss_weight = cfg.locref_loss_weight

    def construct(self, inputs, part_score_targets, part_score_weights,
                  locref_targets, locref_mask,
                  pairwise_targets=None, pairwise_mask=None
                  ):
        """
        Args:
            inputs: input images
            part_score_targets: part score targets
            part_score_weights: part score weights
            locref_targets: location reference targets
            locref_mask: location reference mask
            pairwise_targets: pairwise targets
            pairwise_mask: pairwise mask
        Return:
            total loss
        """
        features, intermediate = self.net(inputs)
        total_loss = self.sce(self.net.part_pred(features), part_score_targets)
        total_loss = self.weight_loss(total_loss, part_score_weights)

        if self.intermediate_supervision:
            part_loss_interm = self.sce(self.net.part_pred_interm(intermediate), part_score_targets)
            part_loss_interm = self.weight_loss(part_loss_interm, part_score_weights)
            total_loss = total_loss + part_loss_interm

        if self.location_refinement:
            locref_pred = self.net.locref(features)
            locref_loss = self.locref_loss_weight * self.locref_loss_func(locref_pred, locref_targets, locref_mask)
            total_loss = total_loss + locref_loss

        if self.pairwise_predict:
            pairwise_pred = self.net.pairwise_pred(features)
            pairwise_loss = self.cfg.pairwise_loss_weight * self.pairwise_loss_func(pairwise_pred, pairwise_targets,
                                                                                    pairwise_mask)
            total_loss = total_loss + pairwise_loss

        return total_loss
