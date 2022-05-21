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
"""Dataloader script."""
import math
import os
import os.path as osp
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from src.utils import build_thresholds
from src.utils import create_anchors_vec
from src.utils import xyxy2xywh


class LoadImages:
    """
    Loader for inference.

    Args:
        path (str): Path to the directory, containing images.
        img_size (list): Size of output image.

    Returns:
        img (np.array): Processed image.
        img0 (np.array): Original image.
    """
    def __init__(self, path, anchor_scales, img_size=(1088, 608)):
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f'Expected a path to the directory with images, got "{path}"')

        self.files = sorted(path.glob('*.jpg'))

        self.anchors, self.strides = create_anchors_vec(anchor_scales)
        self.nf = len(self.files)  # Number of img files.
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nf > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nf:
            raise StopIteration
        img_path = str(self.files[self.count])

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        output = (img, img0)

        return output

    def __getitem__(self, idx):
        idx = idx % self.nf
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        output = (img, img0)

        return output

    def __len__(self):
        return self.nf  # number of files


class LoadVideo:
    """
    Video loader for inference.

    Args:
        path (str): Path to video.
        img_size (tuple): Size of output images size.

    Returns:
        count (int): Number of frame.
        img (np.array): Processed image.
        img0 (np.array): Original image.
    """
    def __init__(self, path, anchor_scales, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.anchors, self.strides = create_anchors_vec(anchor_scales)

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = self.get_size(self.vw, self.vh, self.width, self.height)
        print(f'Lenth of the video: {self.vn:d} frames')

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        _, img0 = self.cap.read()  # BGR
        assert img0 is not None, f'Failed to load frame {self.count:d}'
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        output = (img, img0)

        return output

    def __len__(self):
        return self.vn  # number of files


class JointDataset:
    """
    Loader for all datasets.

    Args:
        root (str): Absolute path to datasets.
        paths (dict): Relative paths for datasets.
        img_size (list): Size of output image.
        augment (bool): Augment images or not.
        transforms: Transform methods.
        config (class): Config with hyperparameters.

    Returns:
        imgs (np_array): Prepared image. Shape (C, H, W)
        tconf (s, m, b) (np_array): Mask with bg (0), gt (1) and ign (-1) indices. Shape (nA, nGh, nGw).
        tbox (s, m, b) (np_array): Targets delta bbox values. Shape (nA, nGh, nGw, 4).
        tid (s, m, b) (np_array): Grid with id for every cell. Shape (nA, nGh, nGw).
    """
    def __init__(
            self,
            root,
            paths,
            img_size=(1088, 608),
            k_max=200,
            augment=False,
            transforms=None,
            config=None,
    ):
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.config = config
        self.anchors, self.strides = create_anchors_vec(config.anchor_scales)
        self.k_max = k_max

        # Iterate for all of datasets to prepare paths to labels
        for ds, img_path in paths.items():
            with open(img_path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        # Search for max pedestrian id in dataset
        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if lb.shape[0] < 1:
                    continue
                if lb.ndim < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for k, v in self.tid_num.items():
            self.tid_start_index[k] = last_index
            last_index += v

        self.nid = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nf = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 40)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nid)
        print('start index')
        print(self.tid_start_index)
        print('=' * 40)

    def get_data(self, img_path, label_path):
        """
        Get and prepare data (augment img).
        """
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError(f'File corrupt {img_path}')
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            s = img_hsv[:, :, 1].astype(np.float32)
            v = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            s *= a
            if a > 1:
                np.clip(s, a_min=0, a_max=255, out=s)

            a = (random.random() * 2 - 1) * fraction + 1
            v *= a
            if a > 1:
                np.clip(v, a_min=0, a_max=255, out=v)

            img_hsv[:, :, 1] = s.astype(np.uint8)
            img_hsv[:, :, 2] = v.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, _ = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        nlbls = len(labels)
        if nlbls > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nlbls > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path

    def __getitem__(self, files_index):
        """
        Iterator function for train dataset
        """
        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c
        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        # Graph mode in Mindspore uses constant shapes
        # Thus, it is necessary to fill targets to max possible ids in image
        to_fill = 100 - labels.shape[0]
        padding = np.zeros((to_fill, 6), dtype=np.float32)
        labels = np.concatenate((labels, padding), axis=0)

        # Calculate confidence mask, bbox delta and ids for every map size
        small, medium, big = build_thresholds(
            labels=labels,
            anchor_vec_s=self.anchors[0],
            anchor_vec_m=self.anchors[1],
            anchor_vec_b=self.anchors[2],
            k_max=self.k_max,
        )

        tconf_s, tbox_s, tid_s, emb_indices_s = small
        tconf_m, tbox_m, tid_m, emb_indices_m = medium
        tconf_b, tbox_b, tid_b, emb_indices_b = big

        total_values = (
            imgs.astype(np.float32),
            tconf_s,
            tbox_s,
            tid_s,
            tconf_m,
            tbox_m,
            tid_m,
            tconf_b,
            tbox_b,
            tid_b,
            emb_indices_s,
            emb_indices_m,
            emb_indices_b,
        )
        return total_values

    def __len__(self):
        return self.nf  # number of batches


class JointDatasetDetection(JointDataset):
    """
    Joint dataset for evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, files_index):
        """
        Iterator function for train dataset.
        """
        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c
        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        targets_size = labels.shape[0]

        # Graph mode in Mindspore uses constant shapes
        # Thus, it is necessary to fill targets to max possible ids in image.
        to_fill = 100 - labels.shape[0]
        padding = np.zeros((to_fill, 6), dtype=np.float32)
        labels = np.concatenate((labels, padding), axis=0)

        output = (imgs.astype(np.float32), labels, targets_size)

        return output


def letterbox(
        img,
        height=608,
        width=1088,
        color=(127.5, 127.5, 127.5),
):
    """
    Resize a rectangular image to a padded rectangular
    and fill padded border with color.
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular

    return img, ratio, dw, dh


def random_affine(
        img,
        targets=None,
        degrees=(-10, 10),
        translate=(.1, .1),
        scale=(.9, 1.1),
        shear=(-2, 2),
        border_value=(127.5, 127.5, 127.5),
):
    """
    Apply several data augmentation techniques,
    such as random rotation, random scale, color jittering
    to reduce overfitting.

    Every rotation and scaling and etc.
    is also applied to targets bbox cords.
    """
    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    r = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    r[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    t = np.eye(3)
    t[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    t[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    s = np.eye(3)
    s[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    s[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    m = s @ t @ r  # Combined rotation matrix. ORDER IS IMPORTANT HERE!
    imw = cv2.warpPerspective(img, m, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=border_value)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if targets.shape[0] > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ m.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

            return imw, targets, m

    return imw
