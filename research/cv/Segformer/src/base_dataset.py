# Copyright 2023 Huawei Technologies Co., Ltd
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
import collections
import random

import cv2
import numpy as np
import mmcv


CropBbox = collections.namedtuple('CropBbox', ['crop_y1', 'crop_y2', 'crop_x1', 'crop_x2'])


class BaseDataset:
    def __init__(self,
                 base_size=(512, 1024),
                 crop_size=(512, 1024),
                 ratio_range=(0.5, 2.0),
                 ignore_label=255,
                 ignore_image=0,
                 cat_max_ratio=0.75,
                 prob=0.5,
                 mean=None,
                 std=None):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.ignore_image = ignore_image
        self.ratio_range = ratio_range
        self.cat_max_ratio = cat_max_ratio
        self.direction = "horizontal"
        self.prob = prob
        self.mean = mean
        self.std = std

    @staticmethod
    def crop_img(img, crop_bbox):
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    @staticmethod
    def photo_metric_distortion(image):
        brightness_delta = 32
        contrast_lower, contrast_upper = (0.5, 1.5)
        saturation_lower, saturation_upper = (0.5, 1.5)
        hue_delta = 18

        def convert(img, alpha=1.0, beta=0.0):
            img = img.astype(np.float32) * alpha + beta
            img = np.clip(img, 0, 255)
            return img.astype(np.uint8)

        def brightness(img):
            if np.random.randint(2):
                return convert(img, beta=random.uniform(
                    -brightness_delta, brightness_delta))
            return img

        def contrast(img):
            if np.random.randint(2):
                return convert(img, alpha=random.uniform(
                    contrast_lower, contrast_upper))
            return img

        def saturation(img):
            if np.random.randint(2):
                img = mmcv.bgr2hsv(img)
                img[:, :, 1] = convert(
                    img[:, :, 1],
                    alpha=random.uniform(saturation_lower, saturation_upper))
                img = mmcv.hsv2bgr(img)
            return img

        def hue(img):
            if np.random.randint(2):
                img = mmcv.bgr2hsv(img)
                img[:, :, 0] = (img[:, :, 0].astype(int) +
                                random.randint(-hue_delta, hue_delta)) % 180
                img = mmcv.hsv2bgr(img)
            return img

        # -----------------------> photo metric distortion <-----------------------
        # random brightness
        image = brightness(image)

        # random contrast first
        mode = np.random.randint(2)
        if mode == 1:
            image = contrast(image)

        # random saturation
        image = saturation(image)
        # random hue
        image = hue(image)
        # random contrast last
        if mode == 0:
            image = contrast(image)

        return image

    @staticmethod
    def default_format_bundle(image, label):
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
        image_out = np.ascontiguousarray(image.transpose((2, 0, 1)))
        label_out = np.array(label).astype('int64')
        return image_out, label_out

    def ratio_resize(self, image, label, is_ratio=True):
        if is_ratio:
            min_ratio, max_ratio = self.ratio_range
            ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
            scale = int(self.base_size[0] * ratio), int(self.base_size[1] * ratio)
        else:
            scale = self.base_size if isinstance(
                self.base_size, tuple) else tuple(self.base_size)
        img, _ = mmcv.imrescale(image, scale, return_scale=True)
        label = mmcv.imrescale(label, scale, interpolation='nearest')
        return img, label

    def get_crop_bbox(self, img):
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
        crop_bbox = CropBbox(crop_y1, crop_y2, crop_x1, crop_x2)
        return crop_bbox

    def crop(self, img, label):
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            for _ in range(10):
                seg_temp = self.crop_img(label, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_label]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)
        img = self.crop_img(img, crop_bbox)
        label = self.crop_img(label, crop_bbox)
        return img, label

    def flip(self, image, label):
        flip = np.random.rand() < self.prob
        if flip:
            image = mmcv.imflip(image, direction=self.direction)
            label = mmcv.imflip(label, direction=self.direction)
        return image, label

    def normalize(self, image):
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        to_rgb = True
        image = mmcv.imnormalize(image, mean, std, to_rgb)
        return image

    def pad(self, image, label):
        padded_img = mmcv.impad(image, shape=tuple(self.crop_size), pad_val=self.ignore_image)
        padded_label = mmcv.impad(label, shape=padded_img.shape[:2], pad_val=self.ignore_label)
        return padded_img, padded_label

    def gen_sample(self, image, label, special_operation=(True, True, True, True, True)):
        is_ratio_resize, is_crop, is_flip, is_photo_distortion, is_pad = special_operation
        # resize
        if is_ratio_resize:
            image, label = self.ratio_resize(image, label)
        else:
            image, label = self.ratio_resize(image, label, False)

        # crop
        if is_crop:
            image, label = self.crop(image, label)

        # flip
        if is_flip:
            image, label = self.flip(image, label)

        # photo metric distortion
        if is_photo_distortion:
            image = self.photo_metric_distortion(image)

        image = self.normalize(image)
        if is_pad:
            image, label = self.pad(image, label)

        image, label = self.default_format_bundle(image, label)
        return image, label

    def get_one_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        scale = self.base_size if isinstance(self.base_size, tuple) else tuple(self.base_size)
        img, _ = mmcv.imrescale(image, scale, return_scale=True)

        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        to_rgb = True
        image = mmcv.imnormalize(img, mean, std, to_rgb)
        image = np.ascontiguousarray(image.transpose((2, 0, 1)))
        return image
