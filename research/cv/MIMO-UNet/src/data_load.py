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
Dataloader
"""

import os

import numpy as np
from PIL import Image

from src.data_augment import PairRandomCrop, PairRandomHorizontalFlip


class DeblurDatasetGenerator:
    """DeblurDatasetGenerator"""
    def __init__(self, image_dir, make_aug=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.random_horizontal_flip = PairRandomHorizontalFlip()
        self.random_crop = PairRandomCrop()
        self.make_aug = make_aug

    def __len__(self):
        """get len"""
        return len(self.image_list)

    def __getitem__(self, idx):
        """get item"""
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))
        image = np.asarray(image)
        label = np.asarray(label)
        if self.make_aug:
            image, label = self.random_horizontal_flip(image, label)
            image, label = self.random_crop(image, label)
        image = image.astype(np.float32) / 255
        label = label.astype(np.float32) / 255

        image = image.transpose(2, 0, 1)  # transform to chw format
        label = label.transpose(2, 0, 1)  # transform to chw format

        return image, label

    @staticmethod
    def _check_image(lst):
        """check image format"""
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError(f"{x} is not .png, .jpeg or .jpg image")


def create_dataset_generator(image_dir, make_aug=False):
    """create dataset generator"""
    dataset_generator = DeblurDatasetGenerator(
        image_dir,
        make_aug=make_aug
    )
    return dataset_generator
