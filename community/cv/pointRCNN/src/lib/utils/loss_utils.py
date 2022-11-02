#!/bin/bash
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
"""loss utils"""
import numpy as np
import mindspore as ms
from mindspore import nn as mnn
from mindspore import ops
from mindspore.ops import functional as mF


class DiceLoss(mnn.Cell):
    """Dice Loss"""
    def __init__(self, ignore_target=-1):
        super(DiceLoss, self).__init__()
        self.ignore_target = ignore_target

    def construct(self, input_, target: ms.Tensor):
        """
        :param input: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input_ = ops.Sigmoid()(input_.view(-1))
        target = target.astype(ms.float32).view(-1)
        mask = (target != self.ignore_target).astype(ms.float32)
        return 1.0 - ms.numpy.sum(
            ops.Minimum()(input_, target) * mask) / ops.clip_by_value(
                ms.numpy.sum(ops.Maximum()
                             (input_, target) * mask), ms.Tensor(1, ms.float32))


class SigmoidFocalClassificationLoss(mnn.Cell):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focuses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def construct(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))

        prediction_probabilities = ops.Sigmoid()(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:

            modulating_factor = ops.Pow()(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha +
                                   (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits: ms.Tensor, labels: ms.Tensor):
    # to be compatible with tensorflow, we don't use ignore_idx

    loss = ops.clip_by_value(
        logits, ms.Tensor(0.0)) - logits * labels.astype(logits.dtype)
    loss += ops.Log1p()(ops.exp(-ops.Abs()(logits)))

    return loss


def get_reg_loss(pred_reg,
                 reg_label,
                 mask,
                 loc_scope,
                 loc_bin_size,
                 num_head_bin,
                 anchor_size):
    return get_reg_loss_ori(pred_reg,
                            reg_label,
                            mask,
                            loc_scope,
                            loc_bin_size,
                            num_head_bin,
                            anchor_size,
                            get_xz_fine=True,
                            get_y_by_bin=False,
                            loc_y_scope=0.5,
                            loc_y_bin_size=0.25,
                            get_ry_fine=False)


def get_reg_loss_ori(pred_reg,
                     reg_label,
                     mask,
                     loc_scope,
                     loc_bin_size,
                     num_head_bin,
                     anchor_size,
                     get_xz_fine=True,
                     get_y_by_bin=False,
                     loc_y_scope=0.5,
                     loc_y_bin_size=0.25,
                     get_ry_fine=False):
    """
    Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.

    :param pred_reg: (N, C)
    :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param num_head_bin: constant
    :param anchor_size: (N, 3) or (3)
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    assert ops.Shape()(pred_reg)[-1] > 0
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    reg_loss_dict = {}
    loc_loss = 0

    # xz localization loss
    x_offset_label, y_offset_label, z_offset_label = reg_label[:,
                                                               0], reg_label[:,
                                                                             1], reg_label[:,
                                                                                           2]
    x_shift = ops.clip_by_value(x_offset_label + loc_scope, ms.Tensor(0.0),
                                ms.Tensor(loc_scope * 2 - 1e-3))
    z_shift = ops.clip_by_value(z_offset_label + loc_scope, ms.Tensor(0.0),
                                ms.Tensor(loc_scope * 2 - 1e-3))

    x_bin_label: ms.Tensor = mF.floor(x_shift / loc_bin_size).astype(
        ms.int32)  # [N,]
    z_bin_label: ms.Tensor = mF.floor(z_shift / loc_bin_size).astype(
        ms.int32)  # [N,]

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r
    cross_entropy = mnn.SoftmaxCrossEntropyWithLogits(True, 'mean')
    op = ops.SoftmaxCrossEntropyWithLogits()
    t1 = pred_reg.gather(ms.numpy.arange(x_bin_l, x_bin_r), 1)
    t2 = pred_reg.gather(ms.numpy.arange(z_bin_l, z_bin_r), 1)
    assert t1.shape == t2.shape

    assert -1 not in ops.Shape()(t1)  # ops.Shape()(t1) == (-1, -1) ??

    loss_x_bin, _ = op(
        t1,
        ops.one_hot(x_bin_label, t1.shape[-1], ms.Tensor(1.0, ms.float32),
                    ms.Tensor(0, ms.float32)))
    loss_z_bin, _ = op(
        t2,
        ops.one_hot(z_bin_label, t2.shape[-1], ms.Tensor(1.0, ms.float32),
                    ms.Tensor(0, ms.float32)))

    loss_x_bin = ops.masked_select(loss_x_bin, mask).mean()
    loss_z_bin = ops.masked_select(loss_z_bin, mask).mean()

    reg_loss_dict['loss_x_bin'] = loss_x_bin.asnumpy()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.asnumpy()
    loc_loss += loss_x_bin + loss_z_bin
    smooth_l1_loss = mnn.SmoothL1Loss(reduction='none')
    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_label = x_shift - (ms.Tensor(x_bin_label, ms.float32) *
                                 loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (ms.Tensor(z_bin_label, ms.float32) *
                                 loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size

        x: ms.Tensor = x_bin_label.view(-1, 1)
        x_bin_onehot = ms.numpy.zeros((x_bin_label.shape[0], per_loc_bin_num))
        x_bin_onehot = ops.functional.tensor_scatter_elements(x_bin_onehot,
                                                              x,
                                                              ms.numpy.ones(
                                                                  x.shape),
                                                              axis=1)
        z = z_bin_label.view(-1, 1)
        z_bin_onehot = ms.numpy.zeros((z_bin_label.shape[0], per_loc_bin_num))
        z_bin_onehot = ops.functional.tensor_scatter_elements(z_bin_onehot,
                                                              z,
                                                              ms.numpy.ones(
                                                                  z.shape),
                                                              axis=1)

        loss_x_res: ms.Tensor = smooth_l1_loss(
            (pred_reg[:, x_res_l:x_res_r] * x_bin_onehot).sum(1),
            x_res_norm_label).masked_select(mask).mean()
        loss_z_res: ms.Tensor = smooth_l1_loss(
            (pred_reg[:, z_res_l:z_res_r] * z_bin_onehot).sum(1),
            z_res_norm_label).masked_select(mask).mean()
        reg_loss_dict['loss_x_res'] = loss_x_res.asnumpy()
        reg_loss_dict['loss_z_res'] = loss_z_res.asnumpy()
        loc_loss += loss_x_res + loss_z_res

    # y localization loss
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_shift = ops.clip_by_value(y_offset_label + loc_y_scope,
                                    ms.Tensor(0.0),
                                    ms.Tensor(loc_y_scope * 2 - 1e-3))
        y_bin_label = (y_shift / loc_y_bin_size).floor().long()
        y_res_label = y_shift - (y_bin_label.float() * loc_y_bin_size +
                                 loc_y_bin_size / 2)
        y_res_norm_label = y_res_label / loc_y_bin_size

        y = y_bin_label.view(-1, 1)
        y_bin_onehot = ms.numpy.zeros((y_bin_label.shape[0], loc_y_bin_num))
        y_bin_onehot = ops.functional.tensor_scatter_elements(y_bin_onehot,
                                                              y,
                                                              ms.numpy.ones(
                                                                  y.shape),
                                                              axis=1)

        loss_y_bin: ms.Tensor = cross_entropy(pred_reg[:, y_bin_l:y_bin_r],
                                              y_bin_label)
        loss_y_res: ms.Tensor = smooth_l1_loss(
            (pred_reg[:, y_res_l:y_res_r] * y_bin_onehot).sum(1),
            y_res_norm_label).masked_select(mask).mean()

        reg_loss_dict['loss_y_bin'] = loss_y_bin.asnumpy()
        reg_loss_dict['loss_y_res'] = loss_y_res.asnumpy()

        loc_loss += loss_y_bin + loss_y_res
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset: ms.Tensor = smooth_l1_loss(
            pred_reg[:, y_offset_l:y_offset_r].sum(1),
            y_offset_label).masked_select(mask).mean()
        reg_loss_dict['loss_y_offset'] = loss_y_offset.asnumpy()
        loc_loss += loss_y_offset

    # angle loss
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_label = reg_label[:, 6]

    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin

        ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (
            2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

        shift_angle = ops.clip_by_value(shift_angle - np.pi * 0.25,
                                        ms.Tensor(1e-3),
                                        ms.Tensor(np.pi * 0.5 -
                                                  1e-3))  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = ops.floor(shift_angle / angle_per_class).astype(
            ms.int32)
        ry_res_label = shift_angle - (ry_bin_label.astype(ms.float32) *
                                      angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = ops.floor(shift_angle / angle_per_class).astype(
            ms.int32)
        ry_res_label = shift_angle - (ry_bin_label.astype(ms.float32) *
                                      angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    ry = ry_bin_label.view(-1, 1)
    ry_bin_onehot = ms.numpy.zeros((ry_bin_label.shape[0], num_head_bin))
    ry_bin_onehot = ops.functional.tensor_scatter_elements(ry_bin_onehot,
                                                           ry,
                                                           ms.numpy.ones(
                                                               ry.shape),
                                                           axis=1)

    loss_ry_bin = cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_res = smooth_l1_loss(
        (pred_reg.gather(ms.numpy.arange(ry_res_l, ry_res_r), 1) *
         ry_bin_onehot).sum(axis=1),
        ry_res_norm_label).masked_select(mask).mean()

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.asnumpy()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.asnumpy()
    angle_loss = loss_ry_bin + loss_ry_res

    # size loss
    size_res_l, size_res_r = ry_res_r - 2 * num_head_bin, ry_res_r + 3 - 2 * num_head_bin
    assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1],
                                                          size_res_r)
    size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg.gather(ms.numpy.arange(size_res_l, size_res_r), 1)
    size_loss = smooth_l1_loss(
        size_res_norm, size_res_norm_label).mean(1).masked_select(mask).mean()

    # Total regression loss
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss

    return loc_loss, angle_loss, size_loss, reg_loss_dict
