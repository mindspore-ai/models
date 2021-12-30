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
"""loss"""
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np

class JointsMSELoss(nn.Cell):
    """JointsMSELoss"""
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight
        self.split = ops.Split(1, 36)
        self.mul = ops.Mul()

    def construct(self, output, target, target_weight):
        """JointsMSELoss construct"""
        shape = output.shape
        batch_size = shape[0]
        num_joints = shape[1]

        heatmaps_pred = self.split(np.reshape(output, (batch_size, num_joints, -1)))
        heatmaps_gt = self.split(np.reshape(target, (batch_size, num_joints, -1)))
        loss = 0

        for idx in range(num_joints):
            heatmaps_pred_idx = np.squeeze(heatmaps_pred[idx])
            heatmaps_gt_idx = np.squeeze(heatmaps_gt[idx])

            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    self.mul(heatmaps_pred_idx, target_weight[:, idx]),
                    self.mul(heatmaps_gt_idx, target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmaps_pred_idx, heatmaps_gt_idx)

        return loss * pow(num_joints, 3)

class NetWithLoss(nn.Cell):
    def __init__(self, network, use_target_weight):
        super(NetWithLoss, self).__init__()
        self.net = network
        self.loss = JointsMSELoss(use_target_weight=use_target_weight)

    def construct(self, _input, target, target_weight):
        output = self.net(_input)

        return self.loss(output, target, target_weight)
