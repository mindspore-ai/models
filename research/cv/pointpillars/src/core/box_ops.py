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
"""box ops mindspore"""

import numpy as np
from mindspore import Tensor
from mindspore import numpy as mnp
from mindspore import ops

from src.core.einsum import einsum


def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for PointPillars in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance, decrease other performance.
        smooth_dim: bool
    """
    # need to convert boxes to z-center format
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    xg, yg, zg, wg, lg, hg, rg = np.split(boxes, 7, axis=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = np.sqrt(la ** 2 + wa ** 2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal

    zt = (zg - za) / ha  # 1.6
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
    if encode_angle_to_vector:
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return np.concatenate([xt, yt, zt, wt, lt, ht, rtx, rty], axis=-1)
    rt = rg - ra
    return np.concatenate([xt, yt, zt, wt, lt, ht, rt], axis=-1)


def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        box_encodings ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance, decrease other performance.
        smooth_dim: bool
    """
    # need to convert box_encodings to z-bottom format
    xa, ya, za, wa, la, ha, ra = ops.Split(axis=-1, output_num=7)(anchors)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = ops.Split(axis=-1, output_num=8)(box_encodings)
    else:
        xt, yt, zt, wt, lt, ht, rt = ops.Split(axis=-1, output_num=7)(box_encodings)
    za = za + ha / 2
    diagonal = ops.Sqrt()(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya

    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:
        lg = ops.Exp()(lt) * la
        wg = ops.Exp()(wt) * wa
        hg = ops.Exp()(ht) * ha
    if encode_angle_to_vector:
        rax = ops.Cos()(ra)
        ray = ops.Sin()(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = ops.Atan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return ops.Concat(axis=-1)([xg, yg, zg, wg, lg, hg, rg])


def bev_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for PointPillars
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, l, w, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance, decrease other performance.
        smooth_dim: bool
    """
    xa, ya, wa, la, ra = ops.Split(axis=-1)(anchors)
    xg, yg, wg, lg, rg = ops.Split(axis=-1)(boxes)
    diagonal = ops.Sqrt()(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
    else:
        lt = ops.Log()(lg / la)
        wt = ops.Log()(wg / wa)
    if encode_angle_to_vector:
        rgx = ops.Cos()(rg)
        rgy = ops.Sin()(rg)
        rax = ops.Cos()(ra)
        ray = ops.Sin()(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return ops.Concat(axis=-1)([xt, yt, wt, lt, rtx, rty])
    rt = rg - ra
    return ops.Concat(axis=-1)([xt, yt, wt, lt, rt])


def bev_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        box_encodings ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance, decrease other performance.
        smooth_dim: bool
    """
    xa, ya, wa, la, ra = ops.Split(axis=-1)(anchors)
    if encode_angle_to_vector:
        xt, yt, wt, lt, rtx, rty = ops.Split(axis=-1)(box_encodings)

    else:
        xt, yt, wt, lt, rt = ops.Split(axis=-1)(box_encodings)

    diagonal = ops.Sqrt()(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
    else:
        lg = ops.Exp()(lt) * la
        wg = ops.Exp()(wt) * wa
    if encode_angle_to_vector:
        rax = ops.Cos()(ra)
        ray = ops.Sin()(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = ops.Atan2()(rgy, rgx)
    else:
        rg = rt + ra
    return ops.Concat(axis=-1)([xg, yg, wg, lg, rg])


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = ops.Stack(axis=1)(
        mnp.unravel_index(mnp.arange(2 ** ndim), (2,) * ndim)
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - Tensor(origin, dtype=dims.dtype)
    corners_norm = Tensor(corners_norm).astype(dims.dtype)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners


def corner_to_standup_nd(boxes_corner):
    """corner to standup nd"""
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(boxes_corner[:, :, i].min(axis=1))
    for i in range(ndim):
        standup_boxes.append(boxes_corner[:, :, i].max(axis=1))
    return ops.Stack(axis=1)(standup_boxes)


def rotation_3d_in_axis(points, angles, axis=0):
    """rotation 3d in axis"""
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = ops.Sin()(angles)
    rot_cos = ops.Cos()(angles)
    ones = ops.OnesLike()(rot_cos)
    zeros = ops.ZerosLike()(rot_cos)
    if axis == 1:
        rot_mat_t = ops.Stack()([
            ops.Stack()([rot_cos, zeros, -rot_sin]),
            ops.Stack()([zeros, ones, zeros]),
            ops.Stack()([rot_sin, zeros, rot_cos])
        ])
    elif axis in (2, -1):
        rot_mat_t = ops.Stack()([
            ops.Stack()([rot_cos, -rot_sin, zeros]),
            ops.Stack()([rot_sin, rot_cos, zeros]),
            ops.Stack()([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_t = ops.Stack()([
            ops.Stack()([zeros, rot_cos, -rot_sin]),
            ops.Stack()([zeros, rot_sin, rot_cos]),
            ops.Stack()([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should in range")
    return einsum('aij,jka->aik', points, rot_mat_t)


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = ops.Sin()(angles)
    rot_cos = ops.Cos()(angles)
    rot_mat_t = ops.Stack()(
        [ops.Stack()([rot_cos, -rot_sin]),
         ops.Stack()([rot_sin, rot_cos])])
    return einsum('aij,jka->aik', points, rot_mat_t)


def center_to_corner_box3d(centers,
                           dims,
                           angles,
                           origin=(0.5, 1.0, 0.5),
                           axis=1):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def project_to_image(points_3d, proj_mat):
    """project to image"""
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    points_4 = ops.Concat(axis=-1)([
        points_3d,
        ops.Zeros()(tuple(points_shape), points_3d.dtype)
    ])
    point_2d = ops.MatMul()(
        ops.Reshape()(points_4, (-1, proj_mat.T.shape[-1])),
        proj_mat.T
    )
    shape = (*points_4.shape[:2], proj_mat.T.shape[-1])
    point_2d = ops.Reshape()(point_2d, shape)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def lidar_to_camera(points, r_rect, velo2cam):
    """lidar to camera"""
    num_points = points.shape[0]
    points = ops.Concat(axis=-1)([points, ops.Ones()((num_points, 1), points.dtype)])
    camera_points = ops.MatMul()(points, ops.MatMul()(r_rect, velo2cam).T)
    return camera_points[..., :3]


def box_lidar_to_camera(data, r_rect, velo2cam):
    """box lidar to camera"""
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return ops.Concat(axis=-1)([xyz, l, h, w, r])
