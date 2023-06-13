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


def image_random_gamma(image, min_gamma=0.7, max_gamma=1.5, clip_image=False):
    gamma = np.random.uniform(min_gamma, max_gamma)
    adjusted = np.power(image, gamma)
    if clip_image:
        np.clip(adjusted, 0.0, 1.0)
    return adjusted


class RandomGamma:
    """define RandomGamma"""

    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    def __call__(self, image):
        return image_random_gamma(
            image, min_gamma=self._min_gamma, max_gamma=self._max_gamma, clip_image=self._clip_image
        )


class TransformChainer:
    """define Transform"""

    def __init__(self, list_of_transforms):
        self._list_of_transforms = list_of_transforms

    def __call__(self, *args):
        list_of_args = list(args)
        for transform in self._list_of_transforms:
            list_of_args = [transform(arg) for arg in list_of_args]
        if len(args) == 1:
            return list_of_args[0]
        return list_of_args


class ConcatTransformSplitChainer:
    """define ConcatTransformSplit"""

    def __init__(self, list_of_transforms):
        self._chainer = TransformChainer(list_of_transforms)

    def __call__(self, *args):
        num_splits = len(args)
        concatenated = np.concatenate(args, axis=0)
        transformed = self._chainer(concatenated)
        split = np.split(transformed, indices_or_sections=num_splits, axis=1)
        return split
