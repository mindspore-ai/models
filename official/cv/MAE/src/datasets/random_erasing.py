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
""" Random Erasing (Cutout)

Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random

import numpy as np


def _get_pixels(per_pixel, rand_color, patch_size, dtype=np.float32):
    """_get_pixels"""
    if per_pixel:
        func = np.random.normal(size=patch_size).astype(dtype)
    elif rand_color:
        func = np.random.normal(size=(patch_size[0], 1, 1)).astype(dtype)
    else:
        func = np.zeros((patch_size[0], 1, 1), dtype=dtype)
    return func


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(self, probability=0.5, min_area=0.02, max_area=1 / 3, min_aspect=0.3,
                 max_aspect=None, mode='const', min_count=1, max_count=None, num_splits=0):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'

    def _erase(self, img, chan, img_h, img_w, dtype):
        """_erase"""
        if random.random() > self.probability:
            pass
        else:
            area = img_h * img_w
            count = self.min_count if self.min_count == self.max_count else \
                random.randint(self.min_count, self.max_count)
            for _ in range(count):
                for _ in range(10):
                    target_area = random.uniform(self.min_area, self.max_area) * area / count
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < img_w and h < img_h:
                        top = random.randint(0, img_h - h)
                        left = random.randint(0, img_w - w)
                        img[:, top:top + h, left:left + w] = _get_pixels(
                            self.per_pixel, self.rand_color, (chan, h, w),
                            dtype=dtype)
                        break
        return img

    def __call__(self, x):
        """RandomErasing apply"""
        if len(x.shape) == 3:
            output = self._erase(x, *x.shape, x.dtype)
        else:
            output = np.zeros_like(x)
            batch_size, chan, img_h, img_w = x.shape
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                output[i] = self._erase(x[i], chan, img_h, img_w, x.dtype)
        return output
