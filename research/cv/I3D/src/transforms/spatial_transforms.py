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
Process pictures spatially.
"""

import random
import cv2


class Compose():

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class CenterCrop():

    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, img):

        img_size = img.shape
        if img_size[0] > img_size[1]:
            max_length = img_size[0] * (256 / img_size[1])
            cv2.resize(img, (int(max_length), 256), interpolation=cv2.INTER_LINEAR)
        elif img_size[0] < img_size[1]:
            max_length = img_size[1] * (256 / img_size[0])
            cv2.resize(img, (256, int(max_length)), interpolation=cv2.INTER_LINEAR)
        elif img_size[0] == img_size[1]:
            cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

        shape = img.shape
        th, tw = self.size
        i = int(round((shape[0] - th) / 2.))
        j = int(round((shape[1] - tw) / 2.))
        return img[i:i + th, j:j + tw]

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip():

    def __call__(self, img):
        if self.p < 0.5:
            return cv2.flip(img, 1)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class RandomCrop():

    def __init__(self, size):
        self.size = (int(size), int(size))

    @staticmethod
    def get_params(img, output_size):

        shape = img.shape
        th, tw = output_size
        if shape[1] == tw and shape[0] == th:
            return 0, 0, shape[0], shape[1]

        i = random.randint(0, shape[0] - th) if shape[0] != th else 0
        j = random.randint(0, shape[1] - tw) if shape[1] != tw else 0
        return i, j, th, tw

    def __call__(self, img):

        img_size = img.shape
        if img_size[0] > img_size[1]:
            max_length = img_size[0] * (256 / img_size[1])
            cv2.resize(img, (int(max_length), 256), interpolation=cv2.INTER_LINEAR)
        elif img_size[0] < img_size[1]:
            max_length = img_size[1] * (256 / img_size[0])
            cv2.resize(img, (256, int(max_length)), interpolation=cv2.INTER_LINEAR)
        elif img_size[0] == img_size[1]:
            cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

        i, j, h, w = self.get_params(img, self.size)
        img = img[i:i + h, j:j + w]
        return img

    def randomize_parameters(self):
        pass
