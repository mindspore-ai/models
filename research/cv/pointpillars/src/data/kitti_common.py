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
"""kitti common"""
import pathlib
from collections import OrderedDict
from concurrent import futures

import numpy as np
from skimage import io


def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes
        add1: bool. add 1 or not

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes
        add1: bool. add 1 or not

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.
        add1: bool. add 1 or not

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union


def get_image_index_str(img_idx):
    """get image index str"""
    return "{:06d}".format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True):
    """get kitti info path"""
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError(f"file not exist: {file_path}")
    if relative_path:
        return str(file_path)
    return str(prefix / file_path)


def get_image_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    """get image path"""
    return get_kitti_info_path(idx, prefix, 'image_2', '.png', training,
                               relative_path, exist_check)


def get_label_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    """get label path"""
    return get_kitti_info_path(idx, prefix, 'label_2', '.txt', training,
                               relative_path, exist_check)


def get_velodyne_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    """get velodyne path"""
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check)


def get_calib_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    """get calib path"""
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check)


def _extend_matrix(mat):
    """extend matrix"""
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """get kitti image info"""
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        """map func"""
        image_info = {'image_idx': idx, 'pointcloud_num_features': 4}
        annotations = None
        if velodyne:
            image_info['velodyne_path'] = get_velodyne_path(idx, path, training, relative_path)
        image_info['img_path'] = get_image_path(idx, path, training, relative_path)

        if with_imageshape:
            img_path = image_info['img_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['img_shape'] = np.array(io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        if calib:
            calib_path = get_calib_path(idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            p0 = np.array(
                [float(info) for info in lines[0].split(' ')[1:13]]
            ).reshape([3, 4])
            p1 = np.array(
                [float(info) for info in lines[1].split(' ')[1:13]]
            ).reshape([3, 4])
            p2 = np.array(
                [float(info) for info in lines[2].split(' ')[1:13]]
            ).reshape([3, 4])
            p3 = np.array(
                [float(info) for info in lines[3].split(' ')[1:13]]
            ).reshape([3, 4])
            if extend_matrix:
                p0 = _extend_matrix(p0)
                p1 = _extend_matrix(p1)
                p2 = _extend_matrix(p2)
                p3 = _extend_matrix(p3)
            image_info['calib/P0'] = p0
            image_info['calib/P1'] = p1
            image_info['calib/P2'] = p2
            image_info['calib/P3'] = p3
            r0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=r0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = r0_rect
            else:
                rect_4x4 = r0_rect
            image_info['calib/R0_rect'] = rect_4x4
            tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                tr_velo_to_cam = _extend_matrix(tr_velo_to_cam)
                tr_imu_to_velo = _extend_matrix(tr_imu_to_velo)
            image_info['calib/Tr_velo_to_cam'] = tr_velo_to_cam
            image_info['calib/Tr_imu_to_velo'] = tr_imu_to_velo
        if annotations is not None:
            image_info['annos'] = annotations
            add_difficulty_to_annos(image_info)
        return image_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)


def get_class_to_label_map():
    """get class to label map"""
    class_to_label = {
        'Car': 0,
        'Pedestrian': 1,
        'Cyclist': 2,
        'Van': 3,
        'Person_sitting': 4,
        'Truck': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': -1,
    }
    return class_to_label


def get_classes():
    """get classes"""
    return get_class_to_label_map().keys()


def remove_dontcare(image_anno):
    """remove dontcare"""
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x != "DontCare"
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][relevant_annotation_indices]
    return img_filtered_annotations


def keep_arrays_by_name(gt_names, used_classes):
    """keep arrays by name"""
    inds = [
        i for i, x in enumerate(gt_names) if x in used_classes
    ]
    inds = np.array(inds, dtype=np.int64)
    return inds


def drop_arrays_by_name(gt_names, used_classes):
    """drop arrays by name"""
    inds = [
        i for i, x in enumerate(gt_names) if x not in used_classes
    ]
    inds = np.array(inds, dtype=np.int64)
    return inds


def kitti_result_line(result_dict, precision=4):
    """kitti result line"""
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError(f"you must specify a value for {key}")
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(f'{val}')
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError(f"unknown key. supported key:{res_dict.keys()}")
    return ' '.join(res_line)


def add_difficulty_to_annos(info):
    """add difficulty to annos"""
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims),), dtype=np.bool)
    moderate_mask = np.ones((len(dims),), dtype=np.bool)
    hard_mask = np.ones((len(dims),), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def get_label_anno(label_path):
    """get label anno"""
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]
    ).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]
    ).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]
    ).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]
    ).reshape(-1)
    if content and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0],))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def get_start_result_anno():
    """get start result anno"""
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations


def empty_result_anno():
    """empty result anno"""
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations


def anno_to_rbboxes(anno):
    """anno to rbboxes"""
    loc = anno["location"]
    dims = anno["dimensions"]
    rots = anno["rotation_y"]
    rbboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    return rbboxes
