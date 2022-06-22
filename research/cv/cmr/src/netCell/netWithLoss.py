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

from mindspore import nn
from mindspore.ops import stop_gradient
from mindspore.ops import operations as P
from mindspore import numpy as np

from src.models.geometric_layers import orthographic_projection


class CustomWithLossCell(nn.Cell):
    """Net with loss function Cell"""
    def __init__(self, smpl, graph_cnn, mesh, smpl_param_regressor, loss_fn):
        super(CustomWithLossCell, self).__init__()
        self.smpl = smpl
        self.graph_cnn = graph_cnn
        self.mesh = mesh
        self.loss = loss_fn
        self.smpl_param_regressor = smpl_param_regressor

        self.concat = P.Concat(axis=-1)

    def construct(self, images, gt_pose, gt_betas, gt_keypoints_3d, gt_keypoints_2d, has_smpl, has_pose_3d):
        """
        Forward propagate of CMR model
        :param images: shape = (B, 3, 224, 224)
        :param gt_pose: shape = (B, 72)
        :param gt_betas: shape = (B, 10)
        :param gt_keypoints_3d: shape = (B, 24, 4)
        :param gt_keypoints_2d: shape = (B, 24, 3)
        :param has_smpl: shape = (B)
        :param has_pose_3d: shape = (B)
        :return: loss value
        """
        gt_vertices = self.smpl(gt_pose, gt_betas)
        batch_size = gt_vertices.shape[0]

        # Returns subsampled mesh and camera parameters
        pred_vertices_sub, pred_camera = self.graph_cnn(images)

        # Upsample mesh in the original size
        pred_vertices = self.mesh(pred_vertices_sub.transpose((0, 2, 1)))

        # Prepare input for SMPL Parameter regressor
        # The input is the predicted and template vertices subsampled by a factor of 4
        # Notice that we detach the GraphCNN
        x = pred_vertices_sub.transpose((0, 2, 1))
        x = self.concat((x, np.broadcast_to(self.mesh.ref_vertices,
                                            (batch_size, self.mesh.ref_vertices.shape[0],\
                                             self.mesh.ref_vertices.shape[1]))))
        x = stop_gradient(x) # (B, 1723, 6)

        # Estimate SMPL parameters and render vertices
        pred_rotmat, pred_shape = self.smpl_param_regressor(x)

        pred_vertices_smpl = self.smpl(pred_rotmat, pred_shape)

        # Get 3D and projected 2D keypoints from the regressed shape
        pred_keypoints_3d = self.smpl.get_joints(pred_vertices)
        pred_keypoints_2d = orthographic_projection(pred_keypoints_3d, pred_camera)[:, :, :2]
        pred_keypoints_3d_smpl = self.smpl.get_joints(pred_vertices_smpl)
        pred_keypoints_2d_smpl = orthographic_projection(pred_keypoints_3d_smpl, stop_gradient(pred_camera))[:, :, :2]

        return self.loss(pred_keypoints_2d, gt_keypoints_2d, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d,
                         pred_vertices, gt_vertices, has_smpl, pred_keypoints_2d_smpl, pred_keypoints_3d_smpl,
                         pred_vertices_smpl, pred_rotmat, pred_shape, gt_pose, gt_betas)
