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
""" utils """
import os
import numpy as np
from PIL import Image


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        """ calculate the result """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    """
    'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    """
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    """ check file dir """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    """ check file dir """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def colorize(gray, palette):
    """ gray: numpy array of the label and 1*3N size list palette """
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color
