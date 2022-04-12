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
"""
Augmentation
"""

import random


class PairRandomCrop:
    """pair random crop"""
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, label):
        def _input_to_factor(img, size):
            """_input_to_factor"""
            img_height, img_width, _ = img.shape
            height, width = size
            if height > img_height or width > img_width:
                raise ValueError(f"Crop size {size} is larger than input image size {(img_height, img_width)}.")

            if width == img_width and height == img_height:
                return 0, 0, img_height, img_width

            top = random.randint(0, img_height - height)
            left = random.randint(0, img_width - width)
            return top, left, height, width

        y, x, h, w = _input_to_factor(image, self.size)
        image, label = image[y:y+h, x:x+w], label[y:y+h, x:x+w]
        assert image.shape == label.shape
        return image, label


class PairRandomHorizontalFlip:
    """pair random horisontal flip"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.prob:
            return img[::, ::-1], label[::, ::-1]
        return img, label
