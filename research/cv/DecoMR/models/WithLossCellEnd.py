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

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import constexpr
import mindspore.numpy as np
from models.grid_sample import grid_sample
from models.geometric_layers import orthographic_projection
from models import SMPL
from models.uv_generator import Index_UV_Generator


@constexpr
def generate_Tensor(temp):
    return Tensor(temp, dtype=mindspore.float32)

def generate_Tensor_Int(temp):
    return Tensor(temp, dtype=mindspore.int32)

class WithLossCellEnd(nn.Cell):
    def __init__(self, DMR, options, uv_weight, tv_factor, auto_prefix=False):
        super(WithLossCellEnd, self).__init__(auto_prefix=False)

        self.DMR = DMR
        self.sampler = Index_UV_Generator(UV_height=options.uv_res, UV_width=-1, uv_type=options.uv_type)
        self.uv_weight = uv_weight
        self.tv_factor = tv_factor

        self.options = options
        self.adaptive_weight = options.adaptive_weight
        self.lam_dp_uv = options.lam_dp_uv
        self.lam_dp_mask = options.lam_dp_mask
        self.lam_uv = options.lam_uv
        self.lam_tv = options.lam_tv
        self.lam_mesh = options.lam_mesh
        self.lam_key3d = options.lam_key3d
        self.lam_key2d = options.lam_key2d
        self.lam_key3d_smpl = options.lam_key3d_smpl
        self.lam_key2d_smpl = options.lam_key2d_smpl
        self.lam_con = options.lam_con
        self.gtkey3d_from_mesh = options.gtkey3d_from_mesh
        self.use_smpl_joints = options.use_smpl_joints

        self.criterion_mask = nn.BCELoss(reduction='mean')
        self.criterion_shape = nn.L1Loss()
        self.criterion_uv = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_keypoints_3d = nn.L1Loss(reduction='none')
        self.criterion_regr = nn.MSELoss()
        self.expand_dims = ops.ExpandDims()
        self.abs = ops.Abs()
        self.sum = ops.ReduceSum()
        self.ones = ops.Ones()
        self.cat2 = ops.Concat(2)
        self.meshgrid = ops.Meshgrid(indexing="ij")
        self.stack = ops.Stack(axis=0)
        self.grid_sample = grid_sample()
        self.smpl = SMPL()
        self.fill = ops.Fill()

    def error_adaptive_weight(self, fit_joint_error):
        weight = (1 - 10 * fit_joint_error)
        weight[weight <= 0] = 0
        return weight

    def dp_loss(self, pred_dp, gt_dp, has_dp, weight=None):

        pred_dp_shape = pred_dp * has_dp.astype(pred_dp.dtype).mean()
        gt_dp_shape = gt_dp * has_dp.astype(pred_dp.dtype).mean()

        if gt_dp_shape.shape[0] > 0:
            gt_dp_shape_temp = self.expand_dims(gt_dp_shape[:, 0], 1)
            gt_mask_shape = gt_dp_shape_temp > 0
            gt_mask_shape = gt_mask_shape.astype(pred_dp.dtype)

            gt_uv_shape = gt_dp_shape[:, 1:]

            pred_mask_shape = self.expand_dims(pred_dp_shape[:, 0], 1)
            pred_uv_shape = pred_dp_shape[:, 1:]

            interpolate_bilinear = ops.ResizeBilinear((gt_dp.shape[2], gt_dp.shape[3]))
            interpolate_nearest = ops.ResizeNearestNeighbor((gt_dp.shape[2], gt_dp.shape[3]))
            pred_mask_shape = interpolate_bilinear(pred_mask_shape)
            pred_uv_shape = interpolate_nearest(pred_uv_shape)

            if weight is not None:
                weight = weight[:, None, None, None] * has_dp.astype(weight.dtype).mean()
            else:
                weight = 1.0

            pred_mask_shape = ops.clip_by_value(pred_mask_shape, 0.0, 1.0)

            loss_mask = self.criterion_mask(pred_mask_shape, gt_mask_shape)
            gt_uv_weight = (gt_uv_shape.abs().max(axis=1, keepdims=True) > 0).astype(pred_dp.dtype)
            weight_ratio = gt_uv_weight.mean(axis=-1).mean(axis=-1)[:, :, None, None] + 1e-8
            gt_uv_weight = gt_uv_weight / weight_ratio

            loss_uv = self.criterion_uv(gt_uv_weight * pred_uv_shape, gt_uv_weight * gt_uv_shape)
            loss_uv = (loss_uv * weight).mean()

            return loss_mask, loss_uv

        return pred_dp.sum() * 0, pred_dp.sum() * 0

    def uv_loss(self, pred_uv_map, gt_uv_map, has_smpl, weight=None):

        uv_weight = self.uv_weight.astype(pred_uv_map.dtype)
        pred_uv_map_shape = pred_uv_map * has_smpl.astype(pred_uv_map.dtype).mean()
        gt_uv_map_with_shape = gt_uv_map * has_smpl.astype(pred_uv_map.dtype).mean()
        if gt_uv_map_with_shape.shape[0] > 0:
            if weight is not None:
                ada_weight = weight[:, None, None, None] * has_smpl.astype(weight.dtype).mean()
            else:
                ada_weight = 1.0
            loss = self.criterion_uv(pred_uv_map_shape * uv_weight, gt_uv_map_with_shape * uv_weight)
            loss = (loss * ada_weight).mean()
            return loss

        return self.fill(mindspore.float32, (1,), 0)

    def tv_loss(self, uv_map):
        tv = self.abs(uv_map[:, 0:-1, 0:-1, :] - uv_map[:, 0:-1, 1:, :]) \
             + self.abs(uv_map[:, 0:-1, 0:-1, :] - uv_map[:, 1:, 0:-1, :])
        return self.sum(tv) / self.tv_factor

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl, weight=None):

        pred_vertices_with_shape = pred_vertices * has_smpl.astype(pred_vertices.type).mean()
        gt_vertices_with_shape = gt_vertices * has_smpl.astype(pred_vertices.type).mean()

        if weight is not None:
            weight = weight[:, None, None] * has_smpl.astype(weight.dtype).mean()
        else:
            weight = 1

        if gt_vertices_with_shape.shape[0] > 0:
            loss = self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
            loss = (loss * weight).mean()
            return loss

        return self.fill(mindspore.float32, (1,), 0)


    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):

        if gt_keypoints_3d.shape[2] == 3:
            tmp = self.ones((gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1), gt_keypoints_3d.dtype)
            gt_keypoints_3d = self.cat2((gt_keypoints_3d, tmp))

        conf = self.expand_dims(gt_keypoints_3d[:, :, -1], -1).copy()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].copy()
        gt_keypoints_3d = gt_keypoints_3d * has_pose_3d.astype(gt_keypoints_3d.dtype).mean()
        conf = conf * has_pose_3d.astype(conf.dtype).mean()

        if weight is not None:
            weight = weight[:, None, None] * has_pose_3d.astype(weight.dtype).mean()
            conf = conf * weight

        pred_keypoints_3d = pred_keypoints_3d * has_pose_3d.astype(pred_keypoints_3d.dtype).mean()
        if gt_keypoints_3d.shape[0] > 0:
            # Align the origin of the first 24 keypoints with the pelvis.
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

            return (conf * self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)).mean()

        return self.fill(mindspore.float32, (1,), 0)

    def smpl_keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):

        if gt_keypoints_3d.shape[2] == 3:
            tmp = self.ones((gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1), gt_keypoints_3d.dtype)
            gt_keypoints_3d = self.cat2((gt_keypoints_3d, tmp))

        conf = self.expand_dims(gt_keypoints_3d[:, :, -1], -1).copy()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].copy()
        gt_keypoints_3d = gt_keypoints_3d * has_pose_3d.astype(gt_keypoints_3d.dtype).mean()
        conf = conf * has_pose_3d.astype(conf.dtype).mean()

        if weight is not None:
            weight = weight[:, None, None] * has_pose_3d.astype(weight.dtype).mean()
            conf = conf * weight

        pred_keypoints_3d = pred_keypoints_3d * has_pose_3d.astype(pred_keypoints_3d.dtype).mean()
        if gt_keypoints_3d.shape[0] > 0:

            gt_root_joint = gt_keypoints_3d[:, 0, :]
            pred_root_joint = pred_keypoints_3d[:, 0, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_root_joint[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_root_joint[:, None, :]

            return (conf * self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)).mean()

        return self.fill(mindspore.float32, (1,), 0)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, weight=None):

        if gt_keypoints_2d.shape[2] == 3:
            conf = self.expand_dims(gt_keypoints_2d[:, :, -1], -1).copy()
        else:
            conf = 1

        if weight is not None:
            weight = weight[:, None, None]
            conf = conf * weight

        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def consistent_loss(self, dp, uv_map, camera, weight=None):

        tmp = np.arange(0, dp.shape[-1], 1) / (dp.shape[-1] - 1)
        tmp = generate_Tensor_Int(tmp)
        tmp = tmp * 2 - 1
        loc_y, loc_x = self.meshgrid((tmp, tmp))

        loc = ops.BroadcastTo((dp.shape[0], -1, -1, -1))(self.stack((loc_x, loc_y)))
        dp_mask = self.expand_dims((dp[:, 0] > 0.5).astype(mindspore.float32), 1)
        loc = dp_mask * loc

        dp_tmp = dp_mask * (dp[:, 1:] * 2 - 1)

        uv_map = uv_map[:, :, :, :-1]
        camera = camera.view(-1, 1, 1, 3)
        uv_map = uv_map + camera[:, :, :, 1:]
        uv_map = uv_map * self.expand_dims(camera[:, :, :, 0], -1)
        warp_loc = self.grid_sample(uv_map.transpose(0, 3, 1, 2), dp_tmp.transpose(0, 2, 3, 1))[:, :2]
        warp_loc = warp_loc * dp_mask

        if weight is not None:
            weight = weight[:, None, None, None]
            dp_mask = dp_mask * weight

        loss_con = nn.MSELoss()(warp_loc * dp_mask, loc * dp_mask)
        return loss_con


    def construct(self, *inputs, **kwargs):

        imges, has_dp, has_smpl, has_pose_3d, has_pose_3d_smpl, gt_dp_iuv, gt_uv_map, gt_vertices, \
        fit_joint_error, gt_keypoints_2d, gt_keypoints_3d, gt_keypoints_2d_smpl, gt_keypoints_3d_smpl = inputs

        pred_dp, pred_uv_map, pred_camera = self.DMR(imges)

        sampled_vertices = self.sampler.resample(pred_uv_map.astype("float32")).astype("float32")

        if self.adaptive_weight:
            ada_weight = self.error_adaptive_weight(fit_joint_error).astype("float32")
        else:
            ada_weight = None

        loss_dp_mask, loss_dp_uv = self.dp_loss(pred_dp, gt_dp_iuv, has_dp, ada_weight)
        loss_dp_mask = loss_dp_mask * self.lam_dp_mask
        loss_dp_uv = loss_dp_uv * self.lam_dp_uv
        CLoss = loss_dp_mask + loss_dp_uv

        loss_uv = self.uv_loss(gt_uv_map.astype("float32"), pred_uv_map.astype("float32"),
                               has_smpl, ada_weight).astype("float32") * self.lam_uv

        loss_tv = 0.0
        if self.lam_tv > 0:
            loss_tv = self.tv_loss(pred_uv_map) * self.lam_tv

        loss_mesh = 0.0
        #loss on mesh
        if self.lam_mesh > 0:
            loss_mesh = self.shape_loss(sampled_vertices, gt_vertices,
                                        has_smpl, ada_weight) * self.lam_mesh

        batch_size = pred_dp.shape[0]
        weight_key = self.ones((batch_size), mindspore.float32)
        if self.gtkey3d_from_mesh:
            if ada_weight is not None:
                weight_key = ada_weight
            has_pose_3d = self.ones((batch_size), mindspore.float32)
            gt_keypoints_3d_mesh = self.smpl.get_train_joints(gt_vertices)
            gt_keypoints_3d_mesh = ops.Concat(-1)((gt_keypoints_3d_mesh,
                                                   self.ones((batch_size, 24, 1), gt_keypoints_3d_mesh.dtype)))
            gt_keypoints_3d = gt_keypoints_3d_mesh


        sampled_joints_3d = self.smpl.get_train_joints(sampled_vertices)
        loss_keypoints_3d = self.keypoint_3d_loss(sampled_joints_3d, gt_keypoints_3d, has_pose_3d, weight_key)
        loss_keypoints_3d = loss_keypoints_3d * self.lam_key3d

        sampled_joints_2d = orthographic_projection(sampled_joints_3d, pred_camera)[:, :, :2]
        loss_keypoints_2d = self.keypoint_loss(sampled_joints_2d, gt_keypoints_2d) * self.lam_key2d

        loss_keypoints_3d_smpl = 0.0
        loss_keypoints_2d_smpl = 0.0
        weight_key_smpl = self.ones((batch_size), mindspore.float32)
        if self.gtkey3d_from_mesh:
            if ada_weight is not None:
                weight_key_smpl = ada_weight
            has_pose_3d = self.ones((batch_size), mindspore.float32)
            gt_keypoints_3d_mesh = self.smpl.get_train_joints(gt_vertices)
            gt_keypoints_3d_mesh = ops.Concat(-1)((gt_keypoints_3d_mesh,
                                                   self.ones((batch_size, 24, 1), gt_keypoints_3d_mesh.dtype)))
            gt_keypoints_3d_smpl = gt_keypoints_3d_mesh

        if self.use_smpl_joints:
            sampled_joints_3d_smpl = self.smpl.get_smpl_joints(sampled_vertices)
            loss_keypoints_3d_smpl = self.smpl_keypoint_3d_loss(sampled_joints_3d_smpl,
                                                                gt_keypoints_3d_smpl, has_pose_3d_smpl, weight_key_smpl)
            loss_keypoints_3d_smpl = loss_keypoints_3d_smpl * self.lam_key3d_smpl

            sampled_joints_2d_smpl = orthographic_projection(sampled_joints_3d_smpl, pred_camera)[:, :, :2]
            loss_keypoints_2d_smpl = self.keypoint_loss(sampled_joints_2d_smpl, gt_keypoints_2d_smpl) \
                                     *self.lam_key2d_smpl

        #consistent loss
        loss_con = 0.0
        if not self.lam_con == 0:
            loss_con = self.consistent_loss(gt_dp_iuv, pred_uv_map, pred_camera, ada_weight) * self.lam_con

        LLoss = loss_uv + loss_tv + loss_keypoints_3d + loss_keypoints_2d + loss_con + \
                loss_keypoints_3d_smpl + loss_keypoints_2d_smpl + loss_tv + loss_mesh

        loss_total = CLoss + LLoss

        return loss_total, CLoss, LLoss
