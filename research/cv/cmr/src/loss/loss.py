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

from mindspore import nn, Tensor
import mindspore
from mindspore import numpy as np
from mindspore import ops

from src.models.geometric_layers import rodrigues

class CustomLoss(nn.Cell):
    """Custom Loss"""
    def __init__(self):
        super(CustomLoss, self).__init__()

        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()

        self.zero = Tensor(0, dtype=mindspore.float32)

    def construct(self, pred_keypoint_2d, gt_keypoint_2d, pred_keypoint_3d, gt_keypoint_3d, has_pose_3d,
                  pred_vertices, gt_vertices, has_smpl, pred_keypoints_2d_smpl, pred_keypoints_3d_smpl,
                  pred_vertices_smpl, pred_rotmat, pred_shape, gt_pose, gt_betas):

        # GraphCNN losses
        loss_keypoints = self.keypoints_loss(pred_keypoint_2d, gt_keypoint_2d)
        loss_keypoints_3d = self.keypoints_3d_loss(pred_keypoint_3d, gt_keypoint_3d, has_pose_3d)
        loss_shape = self.shape_loss(pred_vertices, gt_vertices, has_smpl)

        # SMPL regressor losses
        loss_keypoints_smpl = self.keypoints_loss(pred_keypoints_2d_smpl, gt_keypoint_2d)
        loss_keypoints_3d_smpl = self.keypoints_3d_loss(pred_keypoints_3d_smpl, gt_keypoint_3d, has_pose_3d)
        loss_shape_smpl = self.shape_loss(pred_vertices_smpl, gt_vertices, has_smpl)
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_shape, gt_pose, gt_betas, has_smpl)

        loss = loss_shape_smpl + loss_keypoints_smpl + loss_keypoints_3d_smpl +\
            loss_shape + loss_keypoints + loss_keypoints_3d + 0.1 * loss_regr_betas + loss_regr_pose

        return loss

    def masked_select(self, x, mask):
        """Select masked data from x
        Params:
            x: shape = (B, m, n) or (B, m) or (B, m, n, o)
            mask: 1-d indices
        """
        ndim = x.ndim
        B = x.shape[0]
        m = x.shape[1]
        mask = np.diag(mask)

        x_ = x.reshape(B, -1)
        x_ = ops.matmul(mask, x_)

        if ndim == 4:
            # x shape = (B, m, n, 0)
            n = x.shape[2]
            return x_.reshape(B, m, n, -1)
        if ndim == 2:
            # x shape = (B, m)
            return x_.reshape(B, m)
        # x shape = (B, m, n)
        return x_.reshape(B, m, -1)

    def keypoints_loss(self, pred_keypoint_2d, gt_keypoint_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        Params:
            pred_keypoints_2d: shape = (B, 24, 2)
            gt_keypoint_2d: shape = (B, 24, 3)
        """
        conf = gt_keypoint_2d[:, :, -1][:, :, None].copy()
        loss = (conf * self.criterion_keypoints(pred_keypoint_2d, gt_keypoint_2d[:, :, :-1])).mean()
        return loss

    def keypoints_3d_loss(self, pred_keypoint_3d, gt_keypoint_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        Params:
            pred_keypoint_3d: shape = (B, 24, 3)
            gt_keypoint_3d: shape = (B, 24, 4)
            has_pose_3d: shape = (B)
        """
        has_pose_num = has_pose_3d.sum()

        if has_pose_num > 0:
            conf = gt_keypoint_3d[:, :, -1][:, :, None].copy()
            gt_keypoint_3d = gt_keypoint_3d[:, :, :-1].copy()
            gt_keypoint_3d = self.masked_select(gt_keypoint_3d, has_pose_3d)
            conf = self.masked_select(conf, has_pose_3d)
            pred_keypoint_3d = self.masked_select(pred_keypoint_3d, has_pose_3d)

            gt_pelvis = (gt_keypoint_3d[:, 2, :] + gt_keypoint_3d[:, 3, :]) / 2
            gt_keypoint_3d = gt_keypoint_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoint_3d[:, 2, :] + pred_keypoint_3d[:, 3, :]) / 2
            pred_keypoint_3d = pred_keypoint_3d - pred_pelvis[:, None, :]
            loss = (conf * self.criterion_keypoints(pred_keypoint_3d, gt_keypoint_3d)).mean()
        else:
            loss = Tensor(1.0).fill(0.0)
        return loss

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available.
        Params:
            pred_vertices: shape = (B, 6890, 3)
            gt_vertices: shape = (B, 6890, 3)
            has_smpl: shape = (B)
        """
        has_smpl_num = has_smpl.sum()
        if has_smpl_num > 0:
            pred_vertices_with_shape = self.masked_select(pred_vertices, has_smpl)
            gt_vertices_with_shape = self.masked_select(gt_vertices, has_smpl)

            loss = self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
            loss = loss * len(has_smpl) / has_smpl_num
        else:
            loss = Tensor(1.0).fill(0.0)
        return loss

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available.
        Params:
            pred_rotmat: shape = (B, 24, 3, 3)
            pred_betas: shape = (B, 10)
            gt_pose: shape = (B, 72)
            gt_betas: shape = (B, 10)
            has_smpl: shape = (B)
        """
        has_smpl_num = has_smpl.sum()
        if has_smpl_num > 0:
            pred_rotmat_valid = self.masked_select(pred_rotmat, has_smpl)
            gt_rotmat_valid = rodrigues(self.masked_select(gt_pose, has_smpl).view(-1, 3))
            gt_rotmat_valid = self.masked_select(gt_rotmat_valid.view(-1, 24, 3, 3), has_smpl)

            pred_betas_valid = self.masked_select(pred_betas, has_smpl)
            gt_betas_valid = self.masked_select(gt_betas, has_smpl)

            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)

            loss_regr_pose = loss_regr_pose * len(has_smpl) / has_smpl_num
            loss_regr_betas = loss_regr_betas * len(has_smpl) / has_smpl_num
        else:
            loss_regr_pose = Tensor(1.0).fill(0.0)
            loss_regr_betas = Tensor(1.0).fill(0.0)
        return loss_regr_pose, loss_regr_betas
