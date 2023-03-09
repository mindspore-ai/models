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
# This file was copied from project [sshaoshuai][https://github.com/sshaoshuai/PointRCNN]
"""RPN"""
import importlib
import sys
import os
import mindspore as ms
from mindspore import nn, ops
from mindspore import numpy as np


from src.lib.rpn.proposal_layer import ProposalLayer
from src import layer_utils as pt_utils

import src.lib.utils.loss_utils as loss_utils
from src.lib.config import cfg


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class RPN(nn.Cell):
    """RPN"""
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super(RPN, self).__init__()
        self.training_mode = (mode == 'TRAIN')

        MODEL = importlib.import_module(cfg.RPN.BACKBONE)
        self.backbone_net = MODEL.get_model(input_channels=int(
            cfg.RPN.USE_INTENSITY),
                                            use_xyz=use_xyz)

        # classification branch
        cls_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            cls_layers.append(
                pt_utils.Conv1d(pre_channel,
                                cfg.RPN.CLS_FC[k],
                                bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(p=cfg.RPN.DP_RATIO))
        self.rpn_cls_layer = nn.SequentialCell(cls_layers)

        # regression branch
        per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
        if cfg.RPN.LOC_XZ_FINE:
            reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        reg_channel += 1  # reg y

        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.REG_FC.__len__()):
            reg_layers.append(
                pt_utils.Conv1d(pre_channel,
                                cfg.RPN.REG_FC[k],
                                bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]
        reg_layers.append(
            pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(p=cfg.RPN.DP_RATIO))

        self.rpn_reg_layer = nn.SequentialCell(reg_layers)

        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            self.rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(
                alpha=cfg.RPN.FOCAL_ALPHA[0], gamma=cfg.RPN.FOCAL_GAMMA)
        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':

            self.rpn_cls_loss_func = ops.BinaryCrossEntropy()
        else:
            raise NotImplementedError

        self.proposal_layer = ProposalLayer(mode=mode)
        self.init_weights()

    def init_weights(self):
        """initiate weights"""
        if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            initer = ms.common.initializer.Constant(-np.log((1 - pi) / pi))
            se_par = self.rpn_cls_layer[2].parameters_dict()
            weight_key = [name for name in se_par.keys()
                          if "weight" in name][0]
            data = ms.common.initializer.initializer(initer,
                                                     se_par[weight_key].shape,
                                                     se_par[weight_key].dtype)
            se_par[weight_key].set_data(data)

        initer = ms.common.initializer.Normal(sigma=0.001, mean=0.0)
        all_par = self.rpn_cls_layer[-1].parameters_dict()
        weight_key = [name for name in all_par.keys() if "weight" in name][0]

        data = ms.common.initializer.initializer(initer,
                                                 all_par[weight_key].shape,
                                                 all_par[weight_key].dtype)
        all_par[weight_key].set_data(data)

    def construct(self, input_data):
        """
        :param input_data: dict (point_cloud)
        :return:
        """
        pts_input = input_data['pts_input']
        backbone_xyz, backbone_features = self.backbone_net(
            pts_input)  # (B, N, 3), (B, C, N)

        rpn_cls = self.rpn_cls_layer(backbone_features).swapaxes(
            1, 2)  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(backbone_features).swapaxes(
            1, 2)  # (B, N, C)

        ret_dict = {
            'rpn_cls': rpn_cls,
            'rpn_reg': rpn_reg,
            'backbone_xyz': backbone_xyz,
            'backbone_features': backbone_features
        }

        return ret_dict
