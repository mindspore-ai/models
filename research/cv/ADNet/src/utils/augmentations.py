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
# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_extract_regions.m
# other reference: https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

import numpy as np
import cv2


class ToTensor:
    def __call__(self, cvimage, box=None, action_label=None, conf_label=None):
        return cvimage.astype(np.float32), box, action_label, conf_label


class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, box=None, action_label=None, conf_label=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), box, action_label, conf_label


class CropRegion:
    def __call__(self, image, box, action_label=None, conf_label=None):
        image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * 1.4  # multiplication = 1.4
            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]
        else:
            im = image[:, :, :]

        return im.astype(np.float32), box, action_label, conf_label


# crop "multiplication" times of the box width and height
class CropRegion_withContext:
    def __init__(self, multiplication=None):
        if multiplication is None:
            multiplication = 1.4  # same with default CropRegion
        assert multiplication >= 1, "multiplication should more than 1 so the object itself is not cropped"
        self.multiplication = multiplication

    def __call__(self, image, box, action_label=None, conf_label=None):
        image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * self.multiplication
            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]
        else:
            im = image[:, :, :]

        return im.astype(np.float32), box, action_label, conf_label


class ResizeImage:
    def __init__(self, inputSize):
        self.inputSize = inputSize  # network's input size (which is the output size of this function)

    def __call__(self, image, box, action_label=None, conf_label=None):
        im = cv2.resize(image, dsize=tuple(self.inputSize[:2]))
        return im.astype(np.float32), box, action_label, conf_label


class Compose():
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        # >>> augmentations.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, box=None, action_label=None, conf_label=None):
        for t in self.transforms:
            img, box, action_label, conf_label = t(img, box, action_label, conf_label)
        return img, box, action_label, conf_label


class ADNet_Augmentation:
    def __init__(self, opts):
        self.augment = Compose([
            SubtractMeans(opts['means']),
            CropRegion(),
            ResizeImage(opts['inputSize']),
            # not convert to Tensor,just
            ToTensor()
        ])

    def __call__(self, img, box, action_label=None, conf_label=None):
        return self.augment(img, box, action_label, conf_label)
