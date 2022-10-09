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
"""Proposal target layer"""
import pdb
import mindspore as ms
from mindspore import ops
from mindspore import nn
from mindspore import Tensor
import numpy as np

from src.lib.config import cfg
import src.lib.utils.kitti_utils as kitti_utils
import src.lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import src.lib.utils.iou3d.iou3d_utils as iou3d_utils


class ProposalTargetLayer(nn.Cell):
    """ProposalTargetLayer"""

    def construct(self, **input_dict):
        """construct function"""
        roi_boxes3d, gt_boxes3d = input_dict['roi_boxes3d'], input_dict[
            'gt_boxes3d']

        batch_rois, batch_gt_of_rois, batch_roi_iou = self.sample_rois_for_rcnn(
            roi_boxes3d, gt_boxes3d)

        rpn_xyz, rpn_features = input_dict['rpn_xyz'], input_dict[
            'rpn_features']
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [
                input_dict['rpn_intensity'].expand_dims(axis=2),
                input_dict['seg_mask'].expand_dims(axis=2)
            ]
        else:
            pts_extra_input_list = [input_dict['seg_mask'].expand_dims(axis=2)]

        if cfg.RCNN.USE_DEPTH:
            pts_depth: ms.Tensor = input_dict['pts_depth'] / 70.0 - 0.5

            pts_extra_input_list.append(pts_depth.expand_dims(2))
        pts_extra_input = ops.concat(pts_extra_input_list, 2)

        # point cloud pooling
        pts_feature = ops.concat((pts_extra_input, rpn_features), 2)
        pooled_features, pooled_empty_flag = \
            roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                          sampled_pt_num=cfg.RCNN.NUM_POINTS)

        sampled_pts, sampled_features = pooled_features[:, :, :, 0:
                                                        3], pooled_features[:, :, :,
                                                                            3:]

        # data augmentation
        if cfg.AUG_DATA:
            # data augmentation
            sampled_pts, batch_rois, batch_gt_of_rois = \
                self.data_augmentation(sampled_pts, batch_rois, batch_gt_of_rois)

        # canonical transformation
        batch_size = batch_rois.shape[0]
        roi_ry = batch_rois[:, :, 6] % (2 * ms.numpy.pi)
        roi_center = batch_rois[:, :, 0:3]
        sampled_pts = sampled_pts - roi_center.expand_dims(2)  # (B, M, 512, 3)
        batch_gt_of_rois[:, :, 0:3] = batch_gt_of_rois[:, :, 0:3] - roi_center
        batch_gt_of_rois[:, :, 6] = batch_gt_of_rois[:, :, 6] - roi_ry

        for k in range(batch_size):
            sampled_pts[k] = kitti_utils.rotate_pc_along_y_torch(
                sampled_pts[k], batch_rois[k, :, 6])
            batch_gt_of_rois[k] = kitti_utils.rotate_pc_along_y_torch(
                batch_gt_of_rois[k].expand_dims(1), roi_ry[k]).squeeze(1)

        # regression valid mask
        valid_mask = (pooled_empty_flag == 0)

        reg_valid_mask = (ops.logical_and(
            (batch_roi_iou > cfg.RCNN.REG_FG_THRESH),
            valid_mask)).astype(ms.int32)

        # classification label
        batch_cls_label = (batch_roi_iou > cfg.RCNN.CLS_FG_THRESH).astype(
            ms.int32)

        invalid_mask = ops.logical_and(
            (batch_roi_iou > cfg.RCNN.CLS_BG_THRESH),
            (batch_roi_iou < cfg.RCNN.CLS_FG_THRESH))
        batch_cls_label[valid_mask == 0] = -1
        batch_cls_label[invalid_mask > 0] = -1

        output_dict = {
            'sampled_pts':
            sampled_pts.view(-1, cfg.RCNN.NUM_POINTS, 3),
            'pts_feature':
            sampled_features.view(-1, cfg.RCNN.NUM_POINTS,
                                  sampled_features.shape[3]),
            'cls_label':
            batch_cls_label.view(-1),
            'reg_valid_mask':
            reg_valid_mask.view(-1),
            'gt_of_rois':
            batch_gt_of_rois.view(-1, 7),
            'gt_iou':
            batch_roi_iou.view(-1),
            'roi_boxes3d':
            batch_rois.view(-1, 7)
        }

        return output_dict

    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
        """
        :param roi_boxes3d: (B, M, 7)
        :param gt_boxes3d: (B, N, 8) [x, y, z, h, w, l, ry, cls]
        :return
            batch_rois: (B, N, 7)
            batch_gt_of_rois: (B, N, 8)
            batch_roi_iou: (B, N)
        """
        batch_size = roi_boxes3d.shape[0]

        fg_rois_per_image = int(
            ms.numpy.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))

        batch_rois = ms.numpy.zeros((batch_size, cfg.RCNN.ROI_PER_IMAGE, 7))
        batch_gt_of_rois = ms.numpy.zeros(
            (batch_size, cfg.RCNN.ROI_PER_IMAGE, 7))
        batch_roi_iou = ms.numpy.zeros((batch_size, cfg.RCNN.ROI_PER_IMAGE))

        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]

            k = len(cur_gt) - 1
            while cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]

            # include gt boxes in the candidate rois
            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:,
                                                                0:7])  # (M, N)

            gt_assignment, max_overlaps = ops.ArgMaxWithValue(1)(iou3d)

            # sample fg, easy_bg, hard_bg
            fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
            temp = max_overlaps >= fg_thresh
            fg_inds: ms.Tensor = ops.nonzero(temp).view(-1)

            easy_bg_inds = ops.nonzero(
                (max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO)).view(-1)

            hard_bg_inds = ops.nonzero(
                ops.logical_and(
                    (max_overlaps < cfg.RCNN.CLS_BG_THRESH),
                    (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO))).view(-1)

            fg_num_rois = fg_inds.size  # fg_inds)
            bg_num_rois = hard_bg_inds.size + easy_bg_inds.size

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = ms.Tensor.from_numpy(
                    np.random.permutation(fg_num_rois)).astype(ms.int32)

                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE - fg_rois_per_this_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds,
                                              bg_rois_per_this_image)

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = ms.numpy.floor(
                    np.random.rand(cfg.RCNN.ROI_PER_IMAGE) * fg_num_rois)
                rand_num = ms.Tensor.from_numpy(rand_num).astype(
                    gt_boxes3d.dtype)
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds,
                                              bg_rois_per_this_image)

                fg_rois_per_this_image = 0
            else:
                pdb.set_trace()
                raise NotImplementedError

            # augment the rois by noise
            roi_list, roi_iou_list, roi_gt_list = [], [], []
            if fg_rois_per_this_image > 0:
                fg_rois_src = cur_roi[fg_inds]
                gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
                iou3d_src = max_overlaps[fg_inds]
                fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(
                    fg_rois_src,
                    gt_of_fg_rois,
                    iou3d_src,
                    aug_times=cfg.RCNN.ROI_FG_AUG_TIMES)
                roi_list.append(fg_rois)
                roi_iou_list.append(fg_iou3d)
                roi_gt_list.append(gt_of_fg_rois)

            if bg_rois_per_this_image > 0:
                bg_rois_src = cur_roi[bg_inds]
                gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
                iou3d_src = max_overlaps[bg_inds]
                aug_times = 1 if cfg.RCNN.ROI_FG_AUG_TIMES > 0 else 0
                bg_rois, bg_iou3d = self.aug_roi_by_noise_torch(
                    bg_rois_src, gt_of_bg_rois, iou3d_src, aug_times=aug_times)
                roi_list.append(bg_rois)
                roi_iou_list.append(bg_iou3d)
                roi_gt_list.append(gt_of_bg_rois)

            rois = ops.concat(roi_list, axis=0)
            iou_of_rois = ops.concat(roi_iou_list, axis=0)
            gt_of_rois = ops.concat(roi_gt_list, axis=0)

            batch_rois[idx] = rois
            batch_gt_of_rois[idx] = gt_of_rois
            batch_roi_iou[idx] = iou_of_rois

        return batch_rois, batch_gt_of_rois, batch_roi_iou

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds,
                       bg_rois_per_this_image):
        """sample bg index"""
        randint = ops.UniformInt()
        if hard_bg_inds.size > 0 and easy_bg_inds.size > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image *
                                   cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            minval = ms.Tensor(0, ms.int32)
            maxval = ms.Tensor(hard_bg_inds.size, ms.int32)
            shape = (hard_bg_rois_num,)
            rand_idx = randint(shape, minval, maxval)
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            minval = ms.Tensor(0, ms.int32)
            maxval = ms.Tensor(easy_bg_inds.size, ms.int32)
            shape = (easy_bg_rois_num,)
            rand_idx = randint(shape, minval, maxval)
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = ops.concat([hard_bg_inds, easy_bg_inds], 0)

        elif hard_bg_inds.size > 0 and easy_bg_inds.size == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            minval = ms.Tensor(0, ms.int32)
            maxval = ms.Tensor(hard_bg_inds.size, ms.int32)
            shape = (hard_bg_rois_num,)
            rand_idx = randint(shape, minval, maxval)
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.size == 0 and easy_bg_inds.size > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            minval = ms.Tensor(0, ms.int32)
            maxval = ms.Tensor(easy_bg_inds.size, ms.int32)
            shape = (easy_bg_rois_num,)
            rand_idx = randint(shape, minval, maxval)
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise_torch(self,
                               roi_boxes3d,
                               gt_boxes3d,
                               iou3d_src,
                               aug_times=10):
        """aug roi"""
        iou_of_rois = ms.numpy.zeros((roi_boxes3d.shape[0]),
                                     ms.float32).astype(gt_boxes3d.dtype)
        pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)

        for k in range(roi_boxes3d.shape[0]):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]

            gt_box3d = gt_boxes3d[k].view(1, 7)
            aug_box3d: Tensor = roi_box3d
            keep = True
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                    keep = True
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                    keep = False
                aug_box3d = aug_box3d.view((1, 7))
                iou3d = iou3d_utils.boxes_iou3d_gpu(aug_box3d, gt_box3d)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d.view(-1)
            if cnt == 0 or keep:
                iou_of_rois[k] = iou3d_src[k]
            else:
                iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    @staticmethod
    def random_aug_box3d(box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """
        if cfg.RCNN.REG_AUG_METHOD == 'single':
            rand = ops.UniformReal()
            pos_shift = (rand(3) - 0.5)  # [-0.5 ~ 0.5]
            hwl_scale = (rand(3) - 0.5) / (0.5 / 0.15) + 1.0  #
            angle_rot = (rand(1) - 0.5) / (0.5 /
                                           (np.pi / 12))  # [-pi/12 ~ pi/12]
            aug_box3d = ops.concat([
                box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale,
                box3d[6:7] + angle_rot
            ],
                                   axis=0)
            return aug_box3d
        if cfg.RCNN.REG_AUG_METHOD == 'multiple':
            # pos_range, hwl_range, angle_range, mean_iou
            range_config = [[0.2, 0.1, np.pi / 12, 0.7],
                            [0.3, 0.15, np.pi / 12, 0.6],
                            [0.5, 0.15, np.pi / 9, 0.5],
                            [0.8, 0.15, np.pi / 6, 0.3],
                            [1.0, 0.15, np.pi / 3, 0.2]]
            shape = (1,)
            minval = Tensor(0, ms.int32)
            maxval = Tensor(len(range_config), ms.int32)
            randint = ops.UniformInt(seed=10)
            idx = randint(shape, minval, maxval)
            rand = ops.UniformReal()
            pos_shift = ((rand((3,)) - 0.5) / 0.5) * range_config[idx][0]
            hwl_scale = ((rand(
                (3,)) - 0.5) / 0.5) * range_config[idx][1] + 1.0
            angle_rot = ((rand((1,)) - 0.5) / 0.5) * range_config[idx][2]

            aug_box3d = ops.concat([
                box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale,
                box3d[6:7] + angle_rot
            ],
                                   axis=0)
            return aug_box3d
        rand = ops.UniformReal()
        x_shift = np.random.normal(loc=0, scale=0.3)
        y_shift = np.random.normal(loc=0, scale=0.2)
        z_shift = np.random.normal(loc=0, scale=0.3)
        h_shift = np.random.normal(loc=0, scale=0.25)
        w_shift = np.random.normal(loc=0, scale=0.15)
        l_shift = np.random.normal(loc=0, scale=0.5)
        ry_shift = ((rand - 0.5) / 0.5) * np.pi / 12

        aug_box3d = np.array([
            box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift,
            box3d[3] + h_shift, box3d[4] + w_shift, box3d[5] + l_shift,
            box3d[6] + ry_shift
        ],
                             dtype=np.float32)
        aug_box3d = ms.Tensor.from_numpy(aug_box3d).astype(box3d.dtype)
        return aug_box3d

    def data_augmentation(self, pts, rois, gt_of_rois):
        """
        :param pts: (B, M, 512, 3)
        :param rois: (B, M. 7)
        :param gt_of_rois: (B, M, 7)
        :return:
        """
        batch_size, boxes_num = pts.shape[0], pts.shape[1]
        rand = ops.UniformReal()
        # rotation augmentation
        angles = (rand(
            (batch_size, boxes_num)) - 0.5 / 0.5) * (np.pi / cfg.AUG_ROT_RANGE)

        # calculate gt alpha from gt_of_rois
        temp_x, temp_z, temp_ry = gt_of_rois[:, :,
                                             0], gt_of_rois[:, :,
                                                            2], gt_of_rois[:, :,
                                                                           6]
        temp_beta = ops.atan2(temp_z, temp_x)
        gt_alpha = -ops.Sign()(
            temp_beta) * np.pi / 2 + temp_beta + temp_ry  # (B, M)

        temp_x, temp_z, temp_ry = rois[:, :, 0], rois[:, :, 2], rois[:, :, 6]
        temp_beta = ops.atan2(temp_z, temp_x)
        roi_alpha = -ops.Sign()(
            temp_beta) * np.pi / 2 + temp_beta + temp_ry  # (B, M)

        for k in range(batch_size):
            pts[k] = kitti_utils.rotate_pc_along_y_torch(pts[k], angles[k])
            gt_of_rois[k] = kitti_utils.rotate_pc_along_y_torch(
                gt_of_rois[k].expand_dims(axis=1), angles[k]).squeeze(axis=1)
            rois[k] = kitti_utils.rotate_pc_along_y_torch(
                rois[k].expand_dims(axis=1), angles[k]).squeeze(axis=1)

            # calculate the ry after rotation
            temp_x, temp_z = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2]
            temp_beta = ops.atan2(temp_z, temp_x)
            gt_of_rois[:, :, 6] = ops.Sign()(
                temp_beta) * np.pi / 2 + gt_alpha - temp_beta

            temp_x, temp_z = rois[:, :, 0], rois[:, :, 2]
            temp_beta = ops.atan2(temp_z, temp_x)
            rois[:, :,
                 6] = ops.Sign()(temp_beta) * np.pi / 2 + roi_alpha - temp_beta

        # scaling augmentation
        rand = ops.UniformReal()
        scales = 1 + ((rand((batch_size, boxes_num)) - 0.5) / 0.5) * 0.05
        pts = pts * scales.expand_dims(axis=2).expand_dims(axis=3)
        gt_of_rois[:, :,
                   0:6] = gt_of_rois[:, :, 0:6] * scales.expand_dims(axis=2)
        rois[:, :, 0:6] = rois[:, :, 0:6] * scales.expand_dims(axis=2)

        # flip augmentation
        flip_flag = ops.Sign()(rand((batch_size, boxes_num)) - 0.5)
        pts[:, :, :, 0] = pts[:, :, :, 0] * flip_flag.expand_dims(axis=2)
        gt_of_rois[:, :, 0] = gt_of_rois[:, :, 0] * flip_flag
        # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
        src_ry = gt_of_rois[:, :, 6]
        ry = (flip_flag == 1).astype(
            ms.float32) * src_ry + (flip_flag == -1).astype(
                ms.float32) * (ops.Sign()(src_ry) * ms.numpy.pi - src_ry)
        gt_of_rois[:, :, 6] = ry

        rois[:, :, 0] = rois[:, :, 0] * flip_flag
        # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
        src_ry = rois[:, :, 6]
        ry = (flip_flag == 1).astype(
            ms.float32) * src_ry + (flip_flag == -1).astype(
                ms.float32) * (ops.Sign()(src_ry) * ms.numpy.pi - src_ry)
        rois[:, :, 6] = ry

        return pts, rois, gt_of_rois
