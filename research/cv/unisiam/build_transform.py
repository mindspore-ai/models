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
import numpy as np
import mindspore.dataset.vision.c_transforms as c_transforms
import mindspore.dataset.vision.py_transforms as py_transforms
import mindspore.dataset.transforms as transforms
from rand_augmentation import rand_augmentation
from util import GaussianBlur


def build_transform(input_size):

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(
        translate_const=int(input_size * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        magnitude_std=0.5,
    )

    train_transform = [
        c_transforms.RandomCropDecodeResize(size=input_size, scale=(0.2, 1.)),
        transforms.c_transforms.RandomApply([c_transforms.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)], 0.8),
        py_transforms.ToPIL(),
        py_transforms.RandomGrayscale(0.2),
        transforms.py_transforms.RandomApply([GaussianBlur([.1, 2.])], 0.5),
        rand_augmentation(2, 10, ra_params),
        np.array,
        c_transforms.RandomHorizontalFlip(),
        c_transforms.RandomVerticalFlip(),
        c_transforms.Normalize(mean=(0.485*255, 0.456*255, 0.406*255), std=(0.229*255, 0.224*255, 0.225*255)),
        c_transforms.HWC2CHW(),
        ]

    print('train transform: ', train_transform)
    return train_transform
