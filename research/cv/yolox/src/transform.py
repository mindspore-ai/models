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
# =======================================================================================
""" image transform related """
import random
import math

import cv2
import numpy as np


def get_aug_params(value, center=0):
    if isinstance(value, float):
        min_v = center - value
        max_v = center + value
    elif len(value) == 2:
        min_v = value[0]
        max_v = value[1]
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )
    return random.uniform(min_v, max_v)


def get_affine_matrix(
        target_size,
        degrees=10,
        translate=0.1,
        scales=0.1,
        shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate((corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))).reshape(4, num_gts).T)

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
        img,
        targets=(),
        target_size=(640, 640),
        degrees=10,
        translate=0.1,
        scales=0.1,
        shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    target_length = len(targets)
    if target_length:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)
    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include following 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """ hsv augment """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    """ padding image and transpose dim """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    """ image transform for training """

    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, config=None):
        if config:
            self.max_labels = config.max_gt
            self.flip_prob = config.flip_prob
            self.hsv_prob = config.hsv_prob
            self.strides = config.fpn_strides
            self.input_size = config.input_size
        else:
            self.hsv_prob = 1.0
            self.flip_prob = 0.5
            self.max_labels = max_labels
            self.strides = [8, 16, 32]
            self.input_size = (640, 640)
        self.grid_size = [(self.input_size[0] / x) * (self.input_size[1] / x) for x in
                          self.strides]
        self.num_total_anchor = int(sum(self.grid_size))

    def __call__(self, image, targets, input_dim):
        """ Tran transform call """
        boxes = targets[:, :4]
        labels = targets[:, 4]
        if not boxes.size:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            is_in_boxes_all = np.zeros((self.max_labels, self.num_total_anchor)).astype(np.bool_)
            is_in_boxes_and_center = np.zeros((self.max_labels, self.num_total_anchor)).astype(np.bool_)
            return image, targets, is_in_boxes_all, is_in_boxes_and_center
        image_o = image.copy()
        targets_o = targets.copy()
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        image_t, r_ = preproc(image_t, input_dim)
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if not boxes_t.size:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        true_labels = len(targets_t)

        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        gt_bboxes_per_image = padded_labels[:, 1:5]
        # is_in_boxes_all [gt_max, 8400]
        is_in_boxes_all, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, true_labels)
        # is_in_boxes_all [gt_max, 8400]
        is_in_boxes_all = is_in_boxes_all.any(1).reshape((-1, 1)) * is_in_boxes_all.any(0).reshape((1, -1))
        return image_t, padded_labels, is_in_boxes_all, is_in_boxes_and_center

    def get_grid(self):
        """ get grid in each image """
        grid_size_x = []
        grid_size_y = []
        x_shifts = []  # (1, 6400) (1,1600) (1, 400) -->(1, 8400)
        y_shifts = []  # (1, 6400) (1,1600) (1, 400)
        expanded_strides = []  # (1, 6400) (1,1600) (1, 400)
        for _stride in self.strides:
            grid_size_x.append(int(self.input_size[0] / _stride))
            grid_size_y.append(int(self.input_size[1] / _stride))
        for i in range(len(grid_size_x)):
            xv, yv = np.meshgrid(np.arange(0, grid_size_y[i]), np.arange(0, grid_size_x[i]))
            grid = np.stack((xv, yv), 2).reshape(1, 1, grid_size_x[i], grid_size_y[i], 2)
            grid = grid.reshape(1, -1, 2)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            this_stride = np.zeros((1, grid.shape[1]))
            this_stride.fill(self.strides[i])
            this_stride = this_stride.astype(np.float32)
            expanded_strides.append(this_stride)
        x_shifts = np.concatenate(x_shifts, axis=1)
        y_shifts = np.concatenate(y_shifts, axis=1)
        expanded_strides = np.concatenate(expanded_strides, axis=1)
        return x_shifts, y_shifts, expanded_strides

    def get_in_boxes_info(self, gt_bboxes_per_image, true_labels):
        """ get the pre in-center and in-box info for each image """
        x_shifts, y_shifts, expanded_strides = self.get_grid()
        num_total_anchor = x_shifts.shape[1]
        expanded_strides = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides
        y_shifts_per_image = y_shifts[0] * expanded_strides

        x_centers_per_image = np.expand_dims((x_shifts_per_image + 0.5 * expanded_strides), axis=0)
        x_centers_per_image = np.repeat(x_centers_per_image, self.max_labels, axis=0)

        y_centers_per_image = np.expand_dims((y_shifts_per_image + 0.5 * expanded_strides), axis=0)
        y_centers_per_image = np.repeat(y_centers_per_image, self.max_labels, axis=0)

        gt_bboxes_per_image_l = np.expand_dims((gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]), axis=1)
        gt_bboxes_per_image_l = np.repeat(gt_bboxes_per_image_l, num_total_anchor, axis=1)

        gt_bboxes_per_image_r = np.expand_dims((gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]), axis=1)
        gt_bboxes_per_image_r = np.repeat(gt_bboxes_per_image_r, num_total_anchor, axis=1)

        gt_bboxes_per_image_t = np.expand_dims((gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]), axis=1)
        gt_bboxes_per_image_t = np.repeat(gt_bboxes_per_image_t, num_total_anchor, axis=1)

        gt_bboxes_per_image_b = np.expand_dims((gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]), axis=1)
        gt_bboxes_per_image_b = np.repeat(gt_bboxes_per_image_b, num_total_anchor, axis=1)

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image

        bbox_deltas = np.stack([b_l, b_t, b_r, b_b], 2)
        is_in_boxes = bbox_deltas.min(axis=-1) > 0.0
        is_in_boxes[true_labels:, ...] = False

        center_radius = 2.5
        gt_bboxes_per_image_l = np.repeat(np.expand_dims((gt_bboxes_per_image[:, 0]), 1), num_total_anchor, 1) - \
                                center_radius * np.expand_dims(expanded_strides, 0)

        gt_bboxes_per_image_r = np.repeat(np.expand_dims((gt_bboxes_per_image[:, 0]), 1), num_total_anchor, 1) + \
                                center_radius * np.expand_dims(expanded_strides, 0)

        gt_bboxes_per_image_t = np.repeat(np.expand_dims((gt_bboxes_per_image[:, 1]), 1), num_total_anchor, 1) - \
                                center_radius * np.expand_dims(expanded_strides, 0)

        gt_bboxes_per_image_b = np.repeat(np.expand_dims((gt_bboxes_per_image[:, 1]), 1), num_total_anchor, 1) + \
                                center_radius * np.expand_dims(expanded_strides, 0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image

        center_deltas = np.stack([c_l, c_r, c_t, c_b], 2)
        is_in_centers = center_deltas.min(axis=-1) > 0.0
        is_in_centers[true_labels:, ...] = False  # padding gts are set False

        is_in_boxes_all = is_in_boxes | is_in_centers
        is_in_boxes_and_center = is_in_boxes & is_in_centers
        return is_in_boxes_all, is_in_boxes_and_center


class ValTransform:
    """ image transform for val """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    def __call__(self, img, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy() / 255.0
            img = (img - self.mean) / self.std
        return img, np.zeros((1, 5))


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def statistic_normalize_img(img, statistic_norm):
    """Statistic normalize images."""
    img = np.transpose(img, (1, 2, 0))
    img = img / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if statistic_norm:
        img = (img - mean) / std
    return np.transpose(img, (2, 0, 1)).astype(np.float32)
