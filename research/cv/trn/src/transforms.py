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
"""Implementation of transforms for grouped images"""
import random
from collections import Sized

import numpy as np
from PIL import Image
from mindspore.dataset.transforms.validators import check_random_transform_ops
from mindspore.dataset.vision import Inter
from mindspore.dataset.vision.c_transforms import RandomHorizontalFlip


class GroupRandomHorizontalFlip:
    """Horizontal flip augmentation"""

    def __init__(self, prob=0.5):
        self.worker = RandomHorizontalFlip(prob)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupScale:
    """Rescales the grouped input PIL.Image to the given 'size'.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio.
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

        if not (isinstance(self.size, int) or (isinstance(self.size, Sized) and len(self.size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(self.size))

    def worker(self, img):
        """Resize the input PIL Image to the given size.
        Args:
            img (PIL Image): Image to be resized.

        Returns:
            PIL Image: Resized image.
        """
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)

            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), self.interpolation)

        return img.resize(self.size[::-1], self.interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample:
    def __init__(self, crop_size, scale_size):
        if not isinstance(crop_size, int):
            raise ValueError(f'Crop size must be an integer number, got {type(crop_size)}')
        if not isinstance(crop_size, int):
            raise ValueError(f'Scale size must be an integer number, got {type(scale_size)}')
        self.crop_size = crop_size, crop_size
        self.scale_worker = GroupScale(scale_size)

    def __call__(self, img_group):

        img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for img in img_group:
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)
                flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop:

    def __init__(self, input_size, scales=None, max_distort=1):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Inter.LINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(True, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower right quarter

        return ret


class Stack:

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):

        if self.roll:
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
        return np.concatenate(img_group, axis=2)


class GroupNormalize:
    """Apply normalization to grouped images"""

    def __init__(self, mean, std, channels_num=3, output_type=np.float32):
        self.channels = channels_num

        if len(mean) != channels_num:
            raise ValueError(f'Mean values vector must be of size {channels_num}, got {len(mean)}')
        if len(std) != channels_num:
            raise ValueError(f'Std values vector must be of size {channels_num}, got {len(std)}')

        self.rep_mean = np.array(mean, dtype=output_type)[..., None, None]
        self.rep_std = np.array(std, dtype=output_type)[..., None, None]

    def __call__(self, grouped_images):
        original_shape = grouped_images.shape
        h, w = original_shape[-2:]
        grouped_images = grouped_images.reshape(-1, self.channels, h, w)
        normalized_images = (grouped_images - self.rep_mean) / self.rep_std
        normalized_images = normalized_images.reshape(original_shape)
        return normalized_images


class GroupCompose:
    """Applying several transforms to the grouped images"""

    @check_random_transform_ops
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
