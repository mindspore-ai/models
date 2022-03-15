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
"""Transforms and data augmentation for both image and bbox."""
import random

import cv2
import numpy as np

from src.box_ops import box_xyxy_to_cxcywh


def crop(image, target, region):
    """crop"""
    i, j, h, w = region
    cropped_image = image[i: i + h, j: j + w]

    target = target.copy()
    target["size"] = np.array([h, w])

    fields = ["labels", "area", "iscrowd"]
    if "boxes" in target:
        bboxes = target['boxes']
        max_size = np.array([w, h])
        cropped_boxes = bboxes - np.array([j, i, j, i])
        cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clip(0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(axis=1)
        target['area'] = area
        target['boxes'] = cropped_boxes.reshape(-1, 4)

        keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)

        fields.append("boxes")
        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    """horizontal flip"""
    flipped_image = cv2.flip(image, 1)

    _, w, _ = image.shape

    target = target.copy()
    if "boxes" in target:
        bboxes = target["boxes"]
        bboxes = bboxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) + np.array([w, 0, w, 0])
        target['boxes'] = bboxes

    return flipped_image, target


def resize(image, target, size, max_size=None):
    """resize"""
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """get size with aspect ratio"""
        h, w, _ = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(max_size * min_original_size / max_original_size)

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def get_size(image_size, size, max_size=None):
        """get size"""
        if isinstance(size, (list, tuple)):
            return size[::-1]
        return get_size_with_aspect_ratio(image_size, size, max_size)

    new_h, new_w = get_size(image.shape, size, max_size)
    rescaled_image = cv2.resize(image, (new_w, new_h), cv2.INTER_CUBIC)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig)
        for s, s_orig in zip(rescaled_image.shape[:-1], image.shape[:-1])
    )
    ratio_height, ratio_width = ratios

    target = target.copy()
    if "boxes" in target:
        bboxes = target['boxes']
        scaled_boxes = bboxes * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    target["size"] = np.array([new_h, new_w])

    return rescaled_image, target


class RandomSizeCrop:
    """random size crop"""
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        img_h, img_w, _ = img.shape
        w = random.randint(self.min_size, min(img_w, self.max_size))
        h = random.randint(self.min_size, min(img_h, self.max_size))
        i = np.random.randint(0, img_h - h + 1)
        j = np.random.randint(0, img_w - w + 1)
        region = (i, j, h, w)
        return crop(img, target, region)


class RandomHorizontalFlip:
    """random horizontal flip"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    """random resize"""
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomSelect:
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class Normalize:
    """normalize"""
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        if mean.ndim == 1:
            self.mean = mean.reshape((-1, 1, 1))
        if std.ndim == 1:
            self.std = std.reshape((-1, 1, 1))

    def __call__(self, image, target=None):
        image = (image / 255).transpose(2, 0, 1)
        image = (image - self.mean) / self.std
        if target is None:
            return image, None
        target = target.copy()
        _, h, w = image.shape
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / np.array([w, h, w, h], dtype=np.float32)
            target["boxes"] = boxes
        return image, target


class Compose:
    """compose"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
