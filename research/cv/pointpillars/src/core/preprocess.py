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
"""preprocess"""
import abc
from collections import OrderedDict

import numba
import numpy as np

from src.core import box_np_ops
from src.core.geometry import points_in_convex_polygon_3d_jit
from src.core.geometry import points_in_convex_polygon_jit


class BatchSampler:
    """Batch sampler"""
    def __init__(self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """sample"""
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """reset"""
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """sample"""
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


class DataBasePreprocessing:
    """Data base preprocessing"""
    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractclassmethod
    def _preprocess(cls, db_infos):
        pass


class DBFilterByDifficulty(DataBasePreprocessing):
    """db filter by difficulty"""
    def __init__(self, removed_difficulties):
        self._removed_difficulties = removed_difficulties

    def _preprocess(self, db_infos):
        """preprocess"""
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info["difficulty"] not in self._removed_difficulties
            ]
        return new_db_infos


class DBFilterByMinNumPoint(DataBasePreprocessing):
    """db filter by min num point"""
    def __init__(self, min_gt_point_dict):
        self._min_gt_point_dict = min_gt_point_dict

    def _preprocess(self, db_infos):
        """preprocess"""
        for name, min_num in self._min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos


class DataBasePreprocessor:
    """data base preprocessor"""
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def __call__(self, db_infos):
        for prepor in self._preprocessors:
            db_infos = prepor(db_infos)
        return db_infos


def random_crop_frustum(bboxes,
                        rect,
                        trv2c,
                        p2,
                        max_crop_height=1.0,
                        max_crop_width=0.9):
    """random crop frustum"""
    num_gt = bboxes.shape[0]
    crop_minxy = np.random.uniform(
        [1 - max_crop_width, 1 - max_crop_height],
        [0.3, 0.3],
        size=[num_gt, 2]
    )
    crop_maxxy = np.ones([num_gt, 2], dtype=bboxes.dtype)
    crop_bboxes = np.concatenate([crop_minxy, crop_maxxy], axis=1)
    left = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if left:
        crop_bboxes[:, [0, 2]] -= crop_bboxes[:, 0:1]
    # crop_relative_bboxes to real bboxes
    crop_bboxes *= np.tile(bboxes[:, 2:] - bboxes[:, :2], [1, 2])
    crop_bboxes += np.tile(bboxes[:, :2], [1, 2])
    c, r, t = box_np_ops.projection_matrix_to_crt_kitti(p2)
    frustums = box_np_ops.get_frustum_v2(crop_bboxes, c)
    frustums -= t
    frustums = np.einsum('ij, akj->aki', np.linalg.inv(r), frustums)
    frustums = box_np_ops.camera_to_lidar(frustums, rect, trv2c)

    return frustums


def filter_gt_box_outside_range(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_boxes_bv = box_np_ops.center_to_corner_box2d(
        gt_boxes[:, [0, 1]],
        gt_boxes[:, [3, 3 + 1]],
        gt_boxes[:, 6]
    )
    bounding_box = box_np_ops.minmax_to_corner_2d(np.asarray(limit_range)[np.newaxis, ...])
    ret = points_in_convex_polygon_jit(gt_boxes_bv.reshape(-1, 2), bounding_box)
    return np.any(ret.reshape(-1, 4), axis=1)


def remove_points_in_boxes(points, boxes):
    """remove points in boxes"""
    masks = box_np_ops.points_in_rbbox(points, boxes)
    points = points[np.logical_not(masks.any(-1))]
    return points


def remove_points_outside_boxes(points, boxes):
    """remove points outside boxes"""
    masks = box_np_ops.points_in_rbbox(points, boxes)
    points = points[masks.any(-1)]
    return points


def mask_points_in_corners(points, box_corners):
    """mask points in corners"""
    surfaces = box_np_ops.corner_to_surfaces_3d(box_corners)
    mask = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return mask


@numba.njit
def _rotation_matrix_3d_(rot_mat_t, angle, axis):
    """rotation matrix 3d"""
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_t[:] = np.eye(3)
    if axis == 1:
        rot_mat_t[0, 0] = rot_cos
        rot_mat_t[0, 2] = -rot_sin
        rot_mat_t[2, 0] = rot_sin
        rot_mat_t[2, 2] = rot_cos
    elif axis in (2, -1):
        rot_mat_t[0, 0] = rot_cos
        rot_mat_t[0, 1] = -rot_sin
        rot_mat_t[1, 0] = rot_sin
        rot_mat_t[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_t[1, 1] = rot_cos
        rot_mat_t[1, 2] = -rot_sin
        rot_mat_t[2, 1] = rot_sin
        rot_mat_t[2, 2] = rot_cos


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_t):
    """rotations box 2d jit"""
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_t[0, 0] = rot_cos
    rot_mat_t[0, 1] = -rot_sin
    rot_mat_t[1, 0] = rot_sin
    rot_mat_t[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_t


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    """noise per box"""
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_t)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def noise_per_box_group(boxes, valid_mask, loc_noises, rot_noises, group_nums):
    """noise per box group"""
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_corners[i] = box_corners[i + idx]
                    current_corners[i] -= boxes[i + idx, :2]
                    _rotation_box2d_jit_(current_corners[i], rot_noises[idx + i, j], rot_mat_t)
                    current_corners[i] += boxes[i + idx, :2] + loc_noises[i + idx, j, :2]
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2),
                    box_corners
                )
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx:idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                    break
        idx += num
    return success_mask


@numba.njit
def noise_per_box_group_v2_(boxes, valid_mask, loc_noises, rot_noises,
                            group_nums, global_rot_noises):
    """noise per box group v2"""
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((max_group_num, 2), dtype=boxes.dtype)

    current_grot = np.zeros((max_group_num,), dtype=boxes.dtype)
    dst_grot = np.zeros((max_group_num,), dtype=boxes.dtype)

    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)

    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_box[0, :] = boxes[i + idx]
                    current_radius = np.sqrt(current_box[0, 0] ** 2 + current_box[0, 1] ** 2)
                    current_grot[i] = np.arctan2(current_box[0, 0], current_box[0, 1])

                    dst_grot[i] = current_grot[i] + global_rot_noises[idx + i, j]
                    dst_pos[i, 0] = current_radius * np.sin(dst_grot[i])
                    dst_pos[i, 1] = current_radius * np.cos(dst_grot[i])
                    current_box[0, :2] = dst_pos[i]
                    current_box[0, -1] += (dst_grot[i] - current_grot[i])

                    rot_sin = np.sin(current_box[0, -1])
                    rot_cos = np.cos(current_box[0, -1])
                    rot_mat_t[0, 0] = rot_cos
                    rot_mat_t[0, 1] = -rot_sin
                    rot_mat_t[1, 0] = rot_sin
                    rot_mat_t[1, 1] = rot_cos
                    current_corners[i] = (current_box[0, 2:4] * corners_norm @ rot_mat_t
                                          + current_box[0, :2])
                    current_corners[i] -= current_box[0, :2]

                    _rotation_box2d_jit_(current_corners[i], rot_noises[idx + i, j], rot_mat_t)
                    current_corners[i] += current_box[0, :2] + loc_noises[i + idx, j, :2]
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2),
                    box_corners
                )
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx:idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                        loc_noises[i + idx, j, :2] += (dst_pos[i] - boxes[i + idx, :2])
                        rot_noises[i + idx, j] += (dst_grot[i] - current_grot[i])
                    break
        idx += num
    return success_mask


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises,
                      rot_noises, global_rot_noises):
    """noise per box v2"""
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += (dst_grot - current_grot)

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_t[0, 0] = rot_cos
                rot_mat_t[0, 1] = -rot_sin
                rot_mat_t[1, 0] = rot_sin
                rot_mat_t[1, 1] = rot_cos
                current_corners[:] = (current_box[0, 2:4] * corners_norm @ rot_mat_t
                                      + current_box[0, :2])
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_t)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += (dst_pos - boxes[i, :2])
                    rot_noises[i, j] += (dst_grot - current_grot)
                    break
    return success_mask


@numba.njit
def points_transform_(points, centers, point_masks, loc_transform,
                      rot_transform, valid_mask):
    """points transform"""
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_t = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_t[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_t[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    """box 3d transform"""
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def _select_transform(transform, indices):
    """select transform"""
    result = np.zeros(
        (transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def group_transform_(loc_noise, rot_noise, locs,
                     group_center, valid_mask):
    """group transform"""
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x**2 + y**2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j]) - np.sin(rot_center))
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j]) - np.cos(rot_center))


@numba.njit
def group_transform_v2_(loc_noise, rot_noise, locs,
                        group_center, grot_noise, valid_mask):
    """group transform v2"""
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x**2 + y**2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j] + grot_noise[i, j]) -
                    np.sin(rot_center + grot_noise[i, j])
                )
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j] + grot_noise[i, j]) -
                    np.cos(rot_center + grot_noise[i, j])
                )


def set_group_noise_same_(loc_noise, rot_noise, group_ids):
    """set group noise same"""
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]


def set_group_noise_same_v2_(loc_noise, rot_noise, grot_noise, group_ids):
    """set group noise same v2"""
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]
        grot_noise[i] = grot_noise[gid_to_index_dict[group_ids[i]]]


def get_group_center(locs, group_ids):
    """get group center"""
    num_groups = 0
    group_centers = np.zeros_like(locs)
    group_centers_ret = np.zeros_like(locs)
    group_id_dict = {}
    group_id_num_dict = OrderedDict()
    for i, gid in enumerate(group_ids):
        if gid >= 0:
            if gid in group_id_dict:
                group_centers[group_id_dict[gid]] += locs[i]
                group_id_num_dict[gid] += 1
            else:
                group_id_dict[gid] = num_groups
                num_groups += 1
                group_id_num_dict[gid] = 1
                group_centers[group_id_dict[gid]] = locs[i]
    for i, gid in enumerate(group_ids):
        group_centers_ret[i] = group_centers[group_id_dict[gid]] / group_id_num_dict[gid]
    return group_centers_ret, group_id_num_dict


def noise_per_object(gt_boxes,
                     points=None,
                     valid_mask=None,
                     rotation_perturb=np.pi / 4,
                     center_noise_std=1.0,
                     global_random_rot_range=np.pi / 4,
                     num_try=100,
                     group_ids=None):
    """random rotate or remove each groundtrutn independently"""
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]
    enable_grot = np.abs(global_random_rot_range[0] - global_random_rot_range[1]) >= 1e-3
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0],
        rotation_perturb[1],
        size=[num_boxes, num_try]
    )
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try]
    )
    if group_ids is not None:
        if enable_grot:
            set_group_noise_same_v2_(loc_noises, rot_noises,
                                     global_rot_noises, group_ids)
        else:
            set_group_noise_same_(loc_noises, rot_noises, group_ids)
        group_centers, group_id_num_dict = get_group_center(gt_boxes[:, :3], group_ids)
        if enable_grot:
            group_transform_v2_(loc_noises, rot_noises, gt_boxes[:, :3],
                                group_centers, global_rot_noises, valid_mask)
        else:
            group_transform_(loc_noises, rot_noises, gt_boxes[:, :3],
                             group_centers, valid_mask)
        group_nums = np.array(list(group_id_num_dict.values()), dtype=np.int64)

    origin = [0.5, 0.5, 0]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=origin,
        axis=2
    )
    if group_ids is not None:
        if not enable_grot:
            selected_noise = noise_per_box_group(gt_boxes[:, [0, 1, 3, 4, 6]],
                                                 valid_mask, loc_noises,
                                                 rot_noises, group_nums)
        else:
            selected_noise = noise_per_box_group_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                                     valid_mask, loc_noises,
                                                     rot_noises, group_nums,
                                                     global_rot_noises)
    else:
        if not enable_grot:
            selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises, rot_noises)
        else:
            selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                               valid_mask, loc_noises,
                                               rot_noises, global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                          rot_transforms, valid_mask)

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    """global rotation"""
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(points[:, :3],
                                                            noise_rotation, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(gt_boxes[:, :3],
                                                              noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points


def random_flip(gt_boxes, points, probability=0.5):
    """random flip"""
    enable = np.random.choice(
        [False, True],
        replace=False,
        p=[1 - probability, probability]
    )
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
        points[:, 1] = -points[:, 1]
    return gt_boxes, points


def global_scaling(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    """global scaling"""
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """box collision test"""
    n_boxes = boxes.shape[0]
    k_qboxes = qboxes.shape[0]
    ret = np.zeros((n_boxes, k_qboxes), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(n_boxes):
        for j in range(k_qboxes):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2])
                  - max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3])
                  - max(boxes_standup[i, 1], qboxes_standup[j, 1]))
            if ih > 0 and iw > 0:
                ret[i, j] = _get_ret(lines_boxes, lines_qboxes, i, j)
                if ret[i, j] is False:
                    # now check complete overlap.
                    # box overlap qbox:
                    box_overlap_qbox = _get_box_overlap_another(boxes, qboxes,
                                                                clockwise, i, j)

                    if box_overlap_qbox is False:
                        qbox_overlap_box = _get_box_overlap_another(qboxes, boxes,
                                                                    clockwise, j, i)
                        ret[i, j] = qbox_overlap_box  # if True - collision
                    else:
                        ret[i, j] = True  # collision.
    return ret


@numba.jit(nopython=True)
def _get_ret(lines_boxes, lines_qboxes, i, j):
    """get ret"""
    for k in range(4):
        for l in range(4):
            a = lines_boxes[i, k, 0]
            b = lines_boxes[i, k, 1]
            c = lines_qboxes[j, l, 0]
            d = lines_qboxes[j, l, 1]
            a_c_d = (d[1] - a[1]) * (c[0] - a[0]) > (c[1] - a[1]) * (d[0] - a[0])
            b_c_d = (d[1] - b[1]) * (c[0] - b[0]) > (c[1] - b[1]) * (d[0] - b[0])
            if a_c_d != b_c_d:
                a_b_c = (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
                a_b_d = (d[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (d[0] - a[0])
                if a_b_c != a_b_d:
                    # collision.
                    return True
    return False


@numba.jit(nopython=True)
def _get_box_overlap_another(boxes, qboxes, clockwise, i, j):
    """get box overlap another box"""
    box_overlap_another = True
    for l in range(4):
        for k in range(4):
            vec = boxes[i, k] - boxes[i, (k + 1) % 4]
            if clockwise:
                vec = -vec
            cross = vec[1] * (boxes[i, k, 0] - qboxes[j, l, 0])
            cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, l, 1])
            if cross >= 0:
                box_overlap_another = False
                break
        if box_overlap_another is False:
            break
    return box_overlap_another


def global_translate(gt_boxes, points, noise_translate_std):
    """Apply global translation to gt_boxes and points."""

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])

    noise_translate = np.array([
        np.random.normal(0, noise_translate_std[0], 1),
        np.random.normal(0, noise_translate_std[1], 1),
        np.random.normal(0, noise_translate_std[0], 1)
    ]).T

    points[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

    return gt_boxes, points
