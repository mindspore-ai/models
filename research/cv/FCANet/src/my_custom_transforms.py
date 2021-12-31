# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
custom transforms for our sample
"""
import random
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
import cv2
from src import helpers

########################################[ Function ]########################################


def img_resize_point(img, size):
    """ resize the point from mask to mask"""
    (h, w) = img.shape
    if not isinstance(size, tuple):
        size = (int(w * size), int(h * size))
    M = np.array([[size[0] / w, 0, 0], [0, size[1] / h, 0]])

    pts_y, pts_x = np.where(img == 1)
    pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
    pts_xy_new = np.dot(np.insert(pts_xy, 2, 1, axis=1), M.T).astype(np.int64)

    img_new = np.zeros(size[::-1], dtype=np.uint8)
    for pt in pts_xy_new:
        img_new[pt[1], pt[0]] = 1
    return img_new


########################################[ General ]########################################

# Compose operations
class Compose:
    """ compose multiple transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Transfer:
    """ transfer the data tyle of samples """

    def __init__(self, if_div=True, elems_do=None, elems_undo=None):
        self.if_div = if_div
        if elems_undo is None:
            elems_undo = []
        self.elems_do, self.elems_undo = elems_do, (["meta"] + elems_undo)

    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do is not None and elem not in self.elems_do:
                continue
            if elem in self.elems_undo:
                continue
            tmp = sample[elem]
            tmp = tmp[np.newaxis, :, :] if tmp.ndim == 2 else tmp.transpose((2, 0, 1))
            tmp = tmp / 255 if self.if_div else tmp
            tmp = tmp.astype(np.float32)
            sample[elem] = tmp
        return sample


########################################[ Basic Image Augmentation ]########################################


class RandomFlip:
    """ random flip operation """

    def __init__(
            self, direction=Image.FLIP_LEFT_RIGHT, p=0.5, elems_do=None, elems_undo=None
    ):
        self.direction, self.p = direction, p
        if elems_undo is None:
            elems_undo = []
        self.elems_do, self.elems_undo = elems_do, (["meta"] + elems_undo)

    def __call__(self, sample):
        if random.random() < self.p:
            for elem in sample.keys():
                if self.elems_do is not None and elem not in self.elems_do:
                    continue
                if elem in self.elems_undo:
                    continue
                sample[elem] = np.array(
                    Image.fromarray(sample[elem]).transpose(self.direction)
                )
            sample["meta"]["flip"] = 1
        else:
            sample["meta"]["flip"] = 0
        return sample


class Resize:
    """ resize operation """

    def __init__(
            self, size, mode=None, elems_point=None, elems_do=None, elems_undo=None,
    ):
        self.size, self.mode = size, mode

        if elems_point is None:
            elems_point = ["pos_points_mask", "neg_points_mask"]
        self.elems_point = elems_point

        if elems_undo is None:
            elems_undo = []
        self.elems_do, self.elems_undo = elems_do, (["meta"] + elems_undo)

    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do is not None and elem not in self.elems_do:
                continue
            if elem in self.elems_undo:
                continue

            if elem in self.elems_point:
                sample[elem] = img_resize_point(sample[elem], self.size)
                continue

            if self.mode is None:
                mode = (
                    cv2.INTER_LINEAR
                    if len(sample[elem].shape) == 3
                    else cv2.INTER_NEAREST
                )
            sample[elem] = cv2.resize(sample[elem], self.size, interpolation=mode)
        return sample


class Crop:
    """ crop operation """

    def __init__(self, x_range, y_range, elems_do=None, elems_undo=None):
        self.x_range, self.y_range = x_range, y_range
        if elems_undo is None:
            elems_undo = []
        self.elems_do, self.elems_undo = elems_do, (["meta"] + elems_undo)

    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do is not None and elem not in self.elems_do:
                continue
            if elem in self.elems_undo:
                continue
            sample[elem] = sample[elem][
                self.y_range[0] : self.y_range[1],
                self.x_range[0] : self.x_range[1],
                ...,
            ]

        sample["meta"]["crop_size"] = np.array(
            (self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0])
        )
        sample["meta"]["crop_lt"] = np.array((self.x_range[0], self.y_range[0]))
        return sample


########################################[ Interactive Segmentation ]########################################


class MatchShortSideResize:
    """ resize the samples with short side of fixed size """

    def __init__(self, size, if_must=True, elems_do=None, elems_undo=None):
        self.size, self.if_must = size, if_must
        if elems_undo is None:
            elems_undo = []
        self.elems_do, self.elems_undo = elems_do, (["meta"] + elems_undo)

    def __call__(self, sample):
        src_size = sample["gt"].shape[::-1]

        if (
                (not self.if_must)
                and (src_size[0] >= self.size)
                and (src_size[1] >= self.size)
        ):
            return sample

        src_short_size = min(src_size[0], src_size[1])
        dst_size = (
            int(self.size * src_size[0] / src_short_size),
            int(self.size * src_size[1] / src_short_size),
        )
        assert dst_size[0] == self.size or dst_size[1] == self.size
        Resize(size=dst_size)(sample)
        return sample


class FgContainCrop:
    """ random crop the sample with foreground of at least 1 pixels """

    def __init__(self, crop_size, if_whole=False, elems_do=None, elems_undo=None):
        self.crop_size, self.if_whole = crop_size, if_whole
        if elems_undo is None:
            elems_undo = []
        self.elems_do, self.elems_undo = elems_do, (["meta"] + elems_undo)

    def __call__(self, sample):
        gt = sample["gt"]
        src_size = gt.shape[::-1]
        x_range, y_range = (
            [0, src_size[0] - self.crop_size[0]],
            [0, src_size[1] - self.crop_size[1]],
        )

        if not (gt > 127).any():
            pass
        elif self.if_whole:
            bbox = cv2.boundingRect((gt > 127).astype(np.uint8))

            if bbox[2] <= self.crop_size[0]:
                x_range[1] = min(x_range[1], bbox[0])
                x_range[0] = max(x_range[0], bbox[0] + bbox[2] - self.crop_size[0])
            else:
                x_range = [bbox[0], bbox[0] + bbox[2] - self.crop_size[0]]

            if bbox[3] <= self.crop_size[1]:
                y_range[1] = min(y_range[1], bbox[1])
                y_range[0] = max(y_range[0], bbox[1] + bbox[3] - self.crop_size[1])
            else:
                y_range = [bbox[1], bbox[1] + bbox[3] - self.crop_size[1]]
        else:
            pts_y, pts_x = np.where(gt > 127)
            pts_xy = np.concatenate(
                (pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1
            )
            sp_x, sp_y = pts_xy[random.randint(0, len(pts_xy) - 1)]
            x_range[1], y_range[1] = min(x_range[1], sp_x), min(y_range[1], sp_y)
            x_range[0], y_range[0] = (
                max(x_range[0], sp_x + 1 - self.crop_size[0]),
                max(y_range[0], sp_y + 1 - self.crop_size[1]),
            )

        x_st = random.randint(x_range[0], x_range[1])
        y_st = random.randint(y_range[0], y_range[1])
        Crop(
            x_range=(x_st, x_st + self.crop_size[0]),
            y_range=(y_st, y_st + self.crop_size[1]),
        )(sample)
        return sample


########################################[ Interactive Segmentation (Points) ]########################################


class CatPointMask:
    """  cat the point mask into th input """

    def __init__(self, mode="NO", if_repair=True):
        self.mode, self.if_repair = mode, if_repair

    def __call__(self, sample):
        gt = sample["gt"]

        if "pos_points_mask" in sample.keys() and self.if_repair:
            sample["pos_points_mask"][gt <= 127] = 0
        if "neg_points_mask" in sample.keys() and self.if_repair:
            sample["neg_points_mask"][gt > 127] = 0

        if_gt_empty = not (gt > 127).any()

        if (
                (not if_gt_empty)
                and (not sample["pos_points_mask"].any())
                and self.if_repair
        ):
            if gt[gt.shape[0] // 2, gt.shape[1] // 2] > 127:
                sample["pos_points_mask"][gt.shape[0] // 2, gt.shape[1] // 2] = 1
            else:
                pts_y, pts_x = np.where(gt > 127)
                pts_xy = np.concatenate(
                    (pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1
                )
                pt_pos = pts_xy[random.randint(0, len(pts_xy) - 1)]
                sample["pos_points_mask"][pt_pos[1], pt_pos[0]] = 1

        pos_points_mask, neg_points_mask = (
            sample["pos_points_mask"],
            sample["neg_points_mask"],
        )

        if self.mode == "DISTANCE_POINT_MASK_SRC":
            max_dist = 255
            if if_gt_empty:
                pos_points_mask_dist = np.ones(gt.shape).astype(np.float64) * max_dist
            else:
                pos_points_mask_dist = distance_transform_edt(1 - pos_points_mask)
                pos_points_mask_dist = np.minimum(pos_points_mask_dist, max_dist)

            if not neg_points_mask.any():
                neg_points_mask_dist = np.ones(gt.shape).astype(np.float64) * max_dist
            else:
                neg_points_mask_dist = distance_transform_edt(1 - neg_points_mask)
                neg_points_mask_dist = np.minimum(neg_points_mask_dist, max_dist)

            pos_points_mask_dist, neg_points_mask_dist = (
                pos_points_mask_dist * 255,
                neg_points_mask_dist * 255,
            )
            sample["pos_mask_dist_src"] = pos_points_mask_dist
            sample["neg_mask_dist_src"] = neg_points_mask_dist

        elif self.mode == "DISTANCE_POINT_MASK_FIRST":
            max_dist = 255
            if if_gt_empty:
                pos_points_mask_dist = np.ones(gt.shape).astype(np.float64) * max_dist
            else:
                gt_tmp = (sample["gt"] > 127).astype(np.uint8)
                pred = np.zeros_like(gt_tmp)
                pt, _ = helpers.get_anno_point(pred, gt_tmp, [])
                pos_points_mask = np.zeros_like(gt_tmp)
                pos_points_mask[pt[1], pt[0]] = 1
                pos_points_mask_dist = distance_transform_edt(1 - pos_points_mask)
                pos_points_mask_dist = np.minimum(pos_points_mask_dist, max_dist)
                pos_points_mask_dist = pos_points_mask_dist * 255
            sample["pos_mask_dist_first"] = pos_points_mask_dist
        return sample


class SimulatePoints:
    """  simulate the clicks for training """

    def __init__(self, mode="random", max_point_num=10, if_fixed=False):
        self.mode = mode
        self.max_point_num = max_point_num
        self.if_fixed = if_fixed

    def __call__(self, sample):
        if self.if_fixed:
            object_id = sample["meta"]["id"]
            str_seed = 0
            for c in object_id:
                str_seed += ord(c)
            str_seed = str_seed % 50
            random.seed(str_seed)

        pos_point_num, neg_point_num = random.randint(1, 10), random.randint(0, 10)

        gt = (sample["gt"] > 127).astype(np.uint8)

        if self.mode == "strategy#05":

            pos_points = np.array(
                helpers.get_pos_points_walk(
                    gt, pos_point_num, step=[7, 10, 20], margin=[5, 10, 15, 20]
                )
            )
            neg_points = np.array(
                helpers.get_neg_points_walk(
                    gt,
                    neg_point_num,
                    margin_min=[15, 40, 60],
                    margin_max=[80],
                    step=[10, 15, 25],
                )
            )

            pos_points_mask, neg_points_mask = (
                np.zeros_like(sample["gt"]),
                np.zeros_like(sample["gt"]),
            )
            if list(pos_points):
                pos_points_mask[pos_points[:, 1], pos_points[:, 0]] = 1
            if list(neg_points):
                neg_points_mask[neg_points[:, 1], neg_points[:, 0]] = 1

            sample["pos_points_mask"] = pos_points_mask
            sample["neg_points_mask"] = neg_points_mask

        return sample


current_epoch = 0
record_anno = {}
record_crop_lt = {}
record_if_flip = {}


class ITIS_Crop:
    """  iterative training with crop"""

    def __init__(self, itis_pro=0, mode="random", crop_size=(384, 384)):
        self.itis_pro = itis_pro
        self.mode = mode
        self.crop_size = crop_size

    def __call__(self, sample):
        global current_epoch, record_anno, record_crop_lt, record_if_flip

        object_id = sample["meta"]["id"]
        if (random.random() < self.itis_pro) and current_epoch != 0:
            Crop(
                x_range=(
                    record_crop_lt[object_id][0],
                    record_crop_lt[object_id][0] + self.crop_size[0],
                ),
                y_range=(
                    record_crop_lt[object_id][1],
                    record_crop_lt[object_id][1] + self.crop_size[1],
                ),
            )(sample)
            RandomFlip(p=(1.5 if record_if_flip[object_id] == 1 else -1))(sample)
            sample["pos_points_mask"] = helpers.get_points_mask(
                sample["gt"].shape[::-1], record_anno[object_id][0]
            )
            sample["neg_points_mask"] = helpers.get_points_mask(
                sample["gt"].shape[::-1], record_anno[object_id][1]
            )
        else:
            FgContainCrop(crop_size=self.crop_size, if_whole=False)(sample)
            RandomFlip(p=-1)(sample)
            SimulatePoints(mode=self.mode)(sample)

        return sample


class Decouple:
    """  decouple the sample items for mindspore"""

    def __init__(self, elems=None):
        if elems is None:
            elems = ["img", "gt", "id"]
        self.elems = elems

    def __call__(self, sample):
        return (
            sample["img"],
            sample["gt"],
            sample["pos_points_mask"],
            sample["neg_points_mask"],
            sample["pos_mask_dist_src"],
            sample["neg_mask_dist_src"],
            sample["pos_mask_dist_first"],
            sample["click_loss_weight"],
            sample["first_loss_weight"],
            np.array(sample["meta"]["id_num"], dtype=np.int32),
            np.array(sample["meta"]["crop_lt"]),
            np.array(sample["meta"]["flip"]),
        )


class GeneLossWeight:
    """  generate the loss weight"""

    def __init__(self):
        pass

    def __call__(self, sample):

        pos_dist = sample["pos_mask_dist_src"] / 255.0
        neg_dist = sample["neg_mask_dist_src"] / 255.0
        first_dist = sample["pos_mask_dist_first"] / 255.0
        gt = (sample["gt"] > 127).astype(np.float64)

        tsh, low, high = 100, 0.8, 2.0
        pos_dist = np.minimum(pos_dist, np.ones_like(pos_dist) * tsh)
        neg_dist = np.minimum(neg_dist, np.ones_like(neg_dist) * tsh)
        pos_loss_weight = low + (1.0 - pos_dist / tsh) * (high - low)
        neg_loss_weight = low + (1.0 - neg_dist / tsh) * (high - low)
        pos_loss_weight[gt <= 0.5] = 0
        neg_loss_weight[gt > 0.5] = 0
        click_loss_weight = np.maximum(pos_loss_weight, neg_loss_weight)

        first_dist = np.minimum(first_dist, np.ones_like(first_dist) * tsh)
        first_dist[gt <= 0.5] = tsh
        first_loss_weight = low + (1.0 - first_dist / tsh) * (high - low)

        sample["click_loss_weight"] = click_loss_weight * 255.0
        sample["first_loss_weight"] = first_loss_weight * 255.0

        return sample
