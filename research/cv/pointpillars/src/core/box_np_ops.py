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
"""box np ops"""
import numba
import numpy as np

from src.core.geometry import points_in_convex_polygon_3d_jit


def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for PointPillars in lidar"""
    # need to convert boxes to z-center format
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    xg, yg, zg, wg, lg, hg, rg = np.split(boxes, 7, axis=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = np.sqrt(la ** 2 + wa ** 2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal

    zt = (zg - za) / ha
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
    """box decode for PointPillars in lidar"""
    # need to convert box_encodings to z-bottom format
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = np.split(box_encodings, 8, axis=-1)
    else:
        xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, 7, axis=-1)
    za = za + ha / 2
    diagonal = np.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya

    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:
        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return np.concatenate([xg, yg, zg, wg, lg, hg, rg], axis=-1)


def bev_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for PointPillars in lidar"""
    # need to convert boxes to z-center format
    xa, ya, wa, la, ra = np.split(anchors, 5, axis=-1)
    xg, yg, wg, lg, rg = np.split(boxes, 5, axis=-1)
    diagonal = np.sqrt(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
    if encode_angle_to_vector:
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return np.concatenate([xt, yt, wt, lt, rtx, rty], axis=-1)
    rt = rg - ra
    return np.concatenate([xt, yt, wt, lt, rt], axis=-1)


def bev_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for PointPillars in lidar"""
    # need to convert box_encodings to z-bottom format
    xa, ya, wa, la, ra = np.split(anchors, 5, axis=-1)
    if encode_angle_to_vector:
        xt, yt, wt, lt, rtx, rty = np.split(box_encodings, 6, axis=-1)
    else:
        xt, yt, wt, lt, rt = np.split(box_encodings, 5, axis=-1)
    diagonal = np.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
    else:
        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:
        rg = rt + ra
    return np.concatenate([xg, yg, wg, lg, rg], axis=-1)


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
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
        axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    """corner to standup nd jit"""
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def corner_to_standup_nd(boxes_corner):
    """corner to standup nd"""
    assert len(boxes_corner.shape) == 3
    standup_boxes = [np.min(boxes_corner, axis=1), np.max(boxes_corner, axis=1)]
    return np.concatenate(standup_boxes, -1)


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def rotation_3d_in_axis(points, angles, axis=0):
    """points 3d in axis"""
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_t = np.stack([
            [rot_cos, zeros, -rot_sin],
            [zeros, ones, zeros],
            [rot_sin, zeros, rot_cos]
        ])
    elif axis in (2, -1):
        rot_mat_t = np.stack([
            [rot_cos, -rot_sin, zeros],
            [rot_sin, rot_cos, zeros],
            [zeros, zeros, ones]
        ])
    elif axis == 0:
        rot_mat_t = np.stack([
            [zeros, rot_cos, -rot_sin],
            [zeros, rot_sin, rot_cos],
            [ones, zeros, zeros]
        ])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_t)


def rotation_points_single_angle(points, angle, axis=0):
    """rotation points single angle"""
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_t = np.array(
            [
                [rot_cos, 0, -rot_sin],
                [0, 1, 0],
                [rot_sin, 0, rot_cos]
            ],
            dtype=points.dtype
        )
    elif axis in (2, -1):
        rot_mat_t = np.array(
            [
                [rot_cos, -rot_sin, 0],
                [rot_sin, rot_cos, 0],
                [0, 0, 1]
            ],
            dtype=points.dtype
        )
    elif axis == 0:
        rot_mat_t = np.array(
            [
                [1, 0, 0],
                [0, rot_cos, -rot_sin],
                [0, rot_sin, rot_cos]
            ],
            dtype=points.dtype
        )
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_t


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_t = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_t)


def rotation_box(box_corners, angle):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        box_corners (float array, shape=[N, point_size, 2]): points to be rotated.
        angle (float): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_t = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=box_corners.dtype)
    return box_corners @ rot_mat_t


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
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
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (float)

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
    corners += centers.reshape([-1, 1, 2])
    return corners


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    """box 2d to corner"""
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape((1, 4, 2))
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_t[0, 0] = rot_cos
        rot_mat_t[0, 1] = -rot_sin
        rot_mat_t[1, 0] = rot_sin
        rot_mat_t[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_t + boxes[i, :2]
    return box_corners


def minmax_to_corner_2d(minmax_box):
    """minmax to corner 2d"""
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)


def minmax_to_corner_2d_v2(minmax_box):
    """minmax to corner 2d v2"""
    # N, 4 -> N 4 2
    return minmax_box[..., [0, 1, 0, 3, 2, 3, 2, 1]].reshape(-1, 4, 2)


def center_to_minmax_2d_0_5(centers, dims):
    """center to minmax 2d 0.5"""
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def center_to_minmax_2d(centers, dims, origin=0.5):
    """center to minmax 2d"""
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def limit_period(val, offset=0.5, period=np.pi):
    """limit period"""
    return val - np.floor(val / period + offset) * period


def projection_matrix_to_crt_kitti(proj):
    """projection matrix to CRT kitti"""
    # P = C @ [R|T]
    # C is upper triangular matrix, so we need to inverse CR and use QR
    # stable for all kitti camera projection matrix
    cr = proj[0:3, 0:3]
    ct = proj[0:3, 3]
    rinv_cinv = np.linalg.inv(cr)
    rinv, cinv = np.linalg.qr(rinv_cinv)
    c = np.linalg.inv(cinv)
    r = np.linalg.inv(rinv)
    t = cinv @ ct
    return c, r, t


def get_frustum(bbox_image, c, near_clip=0.001, far_clip=100):
    """get frustum"""
    fku = c[0, 0]
    fkv = -c[1, 1]
    u0v0 = c[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=c.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array([[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
                           dtype=c.dtype)
    near_box_corners = (box_corners - u0v0) / np.array([fku / near_clip, -fkv / near_clip],
                                                       dtype=c.dtype)
    far_box_corners = (box_corners - u0v0) / np.array([fku / far_clip, -fkv / far_clip],
                                                      dtype=c.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def get_frustum_v2(bboxes, c, near_clip=0.001, far_clip=100):
    """get frustum v2"""
    fku = c[0, 0]
    fkv = -c[1, 1]
    u0v0 = c[0:2, 2]
    num_box = bboxes.shape[0]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=c.dtype)[np.newaxis, :, np.newaxis]
    z_points = np.tile(z_points, [num_box, 1, 1])
    box_corners = minmax_to_corner_2d_v2(bboxes)
    near_box_corners = (box_corners - u0v0) / np.array([fku / near_clip, -fkv / near_clip],
                                                       dtype=c.dtype)
    far_box_corners = (box_corners - u0v0) / np.array([fku / far_clip, -fkv / far_clip],
                                                      dtype=c.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=1)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=-1)
    return ret_xyz


def create_anchors_3d_stride(feature_size,
                             sizes=(1.6, 3.9, 1.56),
                             anchor_strides=(0.4, 0.4, 0.0),
                             anchor_offsets=(0.2, -39.8, -1.78),
                             rotations=(0, np.pi / 2),
                             dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz
        anchor_strides: anchor strides
        anchor_offsets: anchor offsets
        rotations: rotations
        dtype: dtype

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    # almost 2x faster than v1
    x_stride, y_stride, z_stride = anchor_strides
    x_offset, y_offset, z_offset = anchor_offsets
    z_centers = np.arange(feature_size[0], dtype=dtype)
    y_centers = np.arange(feature_size[1], dtype=dtype)
    x_centers = np.arange(feature_size[2], dtype=dtype)
    z_centers = z_centers * z_stride + z_offset
    y_centers = y_centers * y_stride + y_offset
    x_centers = x_centers * x_stride + x_offset
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


def create_anchors_3d_range(feature_size,
                            anchor_range,
                            sizes=(1.6, 3.9, 1.56),
                            rotations=(0, np.pi / 2),
                            dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        anchor_range: anchor range
        sizes: [N, 3] list of list or array, size of anchors, xyz
        rotations: rotations
        dtype: dtype

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(
        anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(
        anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(
        anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


def project_to_image(points_3d, proj_mat):
    """project to image"""
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.zeros(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def camera_to_lidar(points, r_rect, velo2cam):
    """camera to lidar"""
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    """lidar to camera"""
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    """box camera to lidar"""
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    """box lidar to camera"""
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)


def remove_outside_points(points, rect, trv2c, p2, image_shape):
    """remove outside points"""
    c, r, t = projection_matrix_to_crt_kitti(p2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, c)
    frustum -= t
    frustum = np.linalg.inv(r) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points


@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    eps: eps
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    n_boxes = boxes.shape[0]
    k_qboxes = query_boxes.shape[0]
    overlaps = np.zeros((n_boxes, k_qboxes), dtype=boxes.dtype)
    for k in range(k_qboxes):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(n_boxes):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = ((boxes[n, 2] - boxes[n, 0] + eps) *
                          (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def points_in_rbbox(points, rbbox, lidar=True):
    """points in rbbox"""
    if lidar:
        h_axis = 2
        origin = [0.5, 0.5, 0]
    else:
        origin = [0.5, 1.0, 0.5]
        h_axis = 1
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6],
                                           origin=origin, axis=h_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


@numba.jit(nopython=False)
def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


@numba.jit(nopython=True)
def sparse_sum_for_anchors_mask(coors, shape):
    """sparse sum for anchors mask"""
    ret = np.zeros(shape, dtype=np.float32)
    for i in range(coors.shape[0]):
        ret[coors[i, 1], coors[i, 2]] += 1
    return ret


@numba.jit(nopython=True)
def fused_get_anchors_area(dense_map, anchors_bv, stride,
                           offset, grid_size):
    """fused get anchors area"""
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)
    grid_size_x = grid_size[0] - 1
    grid_size_y = grid_size[1] - 1
    n = anchors_bv.shape[0]
    ret = np.zeros((n), dtype=dense_map.dtype)
    for i in range(n):
        anchor_coor[0] = np.floor((anchors_bv[i, 0] - offset[0]) / stride[0])
        anchor_coor[1] = np.floor((anchors_bv[i, 1] - offset[1]) / stride[1])
        anchor_coor[2] = np.floor((anchors_bv[i, 2] - offset[0]) / stride[0])
        anchor_coor[3] = np.floor((anchors_bv[i, 3] - offset[1]) / stride[1])
        anchor_coor[0] = max(anchor_coor[0], 0)
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        id_ = dense_map[anchor_coor[3], anchor_coor[2]]
        ia = dense_map[anchor_coor[1], anchor_coor[0]]
        ib = dense_map[anchor_coor[3], anchor_coor[0]]
        ic = dense_map[anchor_coor[1], anchor_coor[2]]
        ret[i] = id_ - ib - ic + ia
    return ret


@numba.jit(nopython=True)
def distance_similarity(points,
                        qpoints,
                        dist_norm,
                        with_rotation=False,
                        rot_alpha=0.5):
    """distance similarity"""
    n_points = points.shape[0]
    k_qpoints = qpoints.shape[0]
    dists = np.zeros((n_points, k_qpoints), dtype=points.dtype)
    rot_alpha_1 = 1 - rot_alpha
    for k in range(k_qpoints):
        for n in range(n_points):
            if np.abs(points[n, 0] - qpoints[k, 0]) <= dist_norm:
                if np.abs(points[n, 1] - qpoints[k, 1]) <= dist_norm:
                    dist = np.sum((points[n, :2] - qpoints[k, :2]) ** 2)
                    dist_normed = min(dist / dist_norm, dist_norm)
                    if with_rotation:
                        dist_rot = np.abs(np.sin(points[n, -1] - qpoints[k, -1]))
                        dists[n, k] = 1 - rot_alpha_1 * dist_normed - rot_alpha * dist_rot
                    else:
                        dists[n, k] = 1 - dist_normed
    return dists


def box3d_to_bbox(box3d, p2):
    """box 3d to bbox"""
    box_corners = center_to_corner_box3d(box3d[:, :3], box3d[:, 3:6],
                                         box3d[:, 6], [0.5, 1.0, 0.5], axis=1)
    box_corners_in_image = project_to_image(box_corners, p2)
    # box_corners_in_image: [N, 8, 2]
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox
