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
"""PointRCNN"""

import mindspore as ms
from mindspore import nn, ops

from src.lib.net.rpn import RPN
from src.lib.net.rcnn_net import RCNNNet
from src.lib.config import cfg


class PointRCNN(nn.Cell):
    """PointRCNN"""
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super(PointRCNN, self).__init__()
        self.training = (mode == 'TRAIN')
        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes,
                                        input_channels=rcnn_input_channels,
                                        use_xyz=use_xyz,
                                        mode=mode)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass
            else:
                raise NotImplementedError

    def construct(self, input_data):
        """construct function"""
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            rpn_output = self.rpn(input_data)
            # print('rpn_cls: ', rpn_output['rpn_cls'].mean())
            output.update(rpn_output)

            # rcnn inference
            if cfg.RCNN.ENABLED:
                rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                backbone_xyz, backbone_features = rpn_output[
                    'backbone_xyz'], rpn_output['backbone_features']
                rpn_scores_raw = rpn_cls[:, :, 0]

                rpn_scores_norm = ops.Sigmoid()(rpn_scores_raw)
                seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).astype(
                    ms.float32)
                pts_depth = ops.norm(backbone_xyz, axis=2, p=2)
                # proposal layer
                rois, roi_scores_raw = self.rpn.proposal_layer(
                    rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
                output['rois'] = rois
                output['roi_scores_raw'] = roi_scores_raw
                output['seg_result'] = seg_mask

                rcnn_input_info = {
                    'rpn_xyz': backbone_xyz,
                    'rpn_features': backbone_features.transpose((0, 2, 1)),
                    'seg_mask': seg_mask,
                    'roi_boxes3d': rois,
                    'pts_depth': pts_depth
                }
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(**rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output
