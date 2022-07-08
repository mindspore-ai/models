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
"""Utils for model"""
import random

import cv2
import mindspore as ms
import mindspore.nn as nn
import numpy as np


def set_global_seed(i):
    """Set global seed"""
    ms.set_seed(i)
    np.random.seed(i)
    random.seed(i)


def image2numpy(image):
    """Transform image to numpy array"""
    image = image.asnumpy()
    image = np.rint(np.clip(np.transpose(image, (1, 2, 0)) * 255, a_min=0, a_max=255)).astype(np.uint8)
    return image


def make_joined_image(im1, im2, im3):
    """Create joined image"""
    im1 = image2numpy(im1)
    im2 = image2numpy(im2)
    im3 = image2numpy(im3)

    height, _, _ = im1.shape
    joined_image = np.zeros((height, height * 3, 3), dtype=np.uint8)
    joined_image[:, :height] = im1
    joined_image[:, height: 2 * height] = im2
    joined_image[:, 2 * height:] = im3
    return joined_image


def save_image(image, image_path):
    """Save image"""
    cv2.imwrite(image_path, image)


def clip_adam_param(beta):
    """Clip Adam betas"""
    return min(max(1e-6, beta), 1 - 1e-6)


def get_lr(initial_lr, lr_policy, num_epochs, num_epochs_decay, dataset_size):
    """
    Learning rate generator.
    For 'linear', we keep the same learning rate for the first <num_epochs>
    epochs and linearly decay the rate to zero over the next
    <num_epochs_decay> epochs.
    """

    if lr_policy == 'linear':
        lrs = [initial_lr] * dataset_size * num_epochs
        for epoch in range(num_epochs_decay):
            lr_epoch = initial_lr * (num_epochs_decay - epoch) / num_epochs_decay
            lrs += [lr_epoch] * dataset_size
        return ms.Tensor(np.array(lrs).astype(np.float32))
    if lr_policy == 'constant':
        return initial_lr
    raise ValueError(f'Unknown lr_policy {lr_policy}')


def enable_batch_statistics(net):
    """Enable batch statistics in all BatchNorms"""
    if isinstance(net, nn.BatchNorm2d):
        net.use_batch_statistics = True
    else:
        for cell in net.cells():
            enable_batch_statistics(cell)
