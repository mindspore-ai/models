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
# ===========================================================================
"""Utilities"""
import random
import numpy as np
import cv2

import mindspore
import mindspore.nn as nn


def fast_hist(predict, label, n):
    """
    fast_hist
    inputs:
        - predict (ndarray)
        - label (ndarray)
        - n (int) - number of classes
    outputs:
        - fast histogram
    """
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(np.int32) + predict[k], minlength=n ** 2).reshape(n, n)


def rescale_batch(inputs, new_scale):
    """
        inputs:
            - inputs (ndarray, shape=(n, c, h, w))
            - new_scale
        outputs: ndarray, shape=(n, c, new_scale[0], new_scale[1])
    """
    n, c, _, _ = inputs.shape
    # n, c, h, w -> n, h, w, c
    input_batch = inputs.transpose((0, 2, 3, 1))
    scaled_batch = np.zeros((n, new_scale[0], new_scale[1], c))
    for i in range(n):
        scaled_batch[i] = cv2.resize(input_batch[i], (new_scale[1], new_scale[0]), interpolation=cv2.INTER_CUBIC)
    scaled_batch = np.ascontiguousarray(scaled_batch)
    # n, h, w, c -> n, c, h, w
    scaled_batch = scaled_batch.transpose((0, 3, 1, 2))
    return scaled_batch


class BuildEvalNetwork(nn.Cell):
    """BuildEvalNetwork"""
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        """construct"""
        output = self.network(input_data)
        output = self.softmax(output)
        return output


def prepare_seed(seed):
    """prepare_seed"""
    mindspore.set_seed(seed)
    random.seed(seed)
