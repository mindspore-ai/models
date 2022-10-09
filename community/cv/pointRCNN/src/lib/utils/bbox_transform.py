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
"""bbox transform"""
import numpy as np
import mindspore as ms
from mindspore import ops


def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    unsqueeze = ops.ExpandDims()
    transpose = ops.Transpose()

    cosa = ops.cos(rot_angle).view(-1, 1)
    sina = ops.sin(rot_angle).view(-1, 1)
    cat = ops.Concat(1)

    raw_1 = cat([cosa, -sina])
    raw_2 = cat([sina, cosa])
    R = cat((unsqueeze(raw_1, 1), unsqueeze(raw_2, 1)))  # (N, 2, 2)
    pc_temp = unsqueeze(pc[:, [0, 2]], 1)  # (N, 1, 2)

    pc[:, [0, 2]] = ops.matmul(pc_temp, transpose(R, (0, 2, 1))).squeeze(1)
    return pc


def decode_bbox_target(roi_box3d,
                       pred_reg,
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
    :param roi_box3d: (N, 7)
    :param pred_reg: (N, C)
    :param loc_scope:
    :param loc_bin_size:
    :param num_head_bin:
    :param anchor_size:
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    # recover xz localization
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    argmax = ops.Argmax(1)

    x_bin: ms.Tensor = argmax(pred_reg[:, x_bin_l:x_bin_r])
    z_bin: ms.Tensor = argmax(pred_reg[:, z_bin_l:z_bin_r])

    pos_x = x_bin.astype(
        ms.float32) * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_z = z_bin.astype(
        ms.float32) * loc_bin_size + loc_bin_size / 2 - loc_scope

    unsqueeze = ops.ExpandDims()
    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_norm = ops.gather_elements(pred_reg[:, x_res_l:x_res_r], 1,
                                         unsqueeze(x_bin, 1)).squeeze(axis=1)
        z_res_norm = ops.gather_elements(pred_reg[:, z_res_l:z_res_r], 1,
                                         unsqueeze(z_bin, 1)).squeeze(axis=1)
        x_res = x_res_norm * loc_bin_size
        z_res = z_res_norm * loc_bin_size

        pos_x += x_res
        pos_z += z_res

    # recover y localization
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_bin = argmax(pred_reg[:, y_bin_l:y_bin_r], dim=1)
        y_res_norm = ops.gather_elements(pred_reg[:, y_res_l:y_res_r],
                                         dim=1,
                                         index=unsqueeze(y_bin,
                                                         1)).squeeze(axis=1)
        y_res = y_res_norm * loc_y_bin_size
        pos_y = y_bin.astype(
            ms.float32
        ) * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
        pos_y = pos_y + roi_box3d[:, 1]
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]

    # recover ry rotation
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
    argmax = ops.Argmax(1)
    ry_bin = argmax(pred_reg[:, ry_bin_l:ry_bin_r])
    ry_res_norm = ops.gather_elements(pred_reg[:, ry_res_l:ry_res_r],
                                      dim=1,
                                      index=unsqueeze(ry_bin,
                                                      1)).squeeze(axis=1)
    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (ry_bin.astype(ms.float32) * angle_per_class +
              angle_per_class / 2) + ry_res - np.pi / 4
    else:
        angle_per_class = (2 * np.pi) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)

        # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
        ry = (ry_bin.astype(ms.float32) * angle_per_class + ry_res) % (2 *
                                                                       np.pi)
        ry = ops.select(ry > np.pi, ry - 2 * np.pi, ry)
        # ry[ry > np.pi] -= 2 * np.pi

    # recover size
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]

    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size

    # shift to original coords
    roi_center = roi_box3d[:, 0:3]
    cat = ops.Concat(1)
    shift_ret_box3d = cat((pos_x.view(-1, 1), pos_y.view(-1, 1),
                           pos_z.view(-1, 1), hwl, ry.view(-1, 1)))
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = roi_box3d[:, 6]
        ret_box3d = rotate_pc_along_y_torch(shift_ret_box3d, -roi_ry)
        ret_box3d[:, 6] += roi_ry
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]

    return ret_box3d
