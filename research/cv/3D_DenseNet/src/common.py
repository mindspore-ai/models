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
"""
common parameters and image processing function
"""
import math
import numpy as np
import cv2

data_dm = 2
ignore_label = 9
num_classes = 4
crop_size = (64, 64, 64)
"""Note"""
checkpoint_name = 'model_3d_denseseg_v1'
num_checkpoint = '20000'
note = str(num_checkpoint) + '_' + checkpoint_name

def make_one_hot(labels):
    """
    Converts an integer label  to a one-hot Variable.
    Parameters
    ----------
    labels : N x 1 x D x H x W, where N is batch size.
             Each value is an integer representing correct classification.
    C : integer number of classes in labels.
    Returns
    -------
    target : N x C x D x H x W, where C is class number. One-hot encoded.
    """
    labels_extend = labels.clone()
    labels_extend.unsqueeze_(1)
    one_hot.scatter_(1, labels_extend, 1) #Copy 1 to one_hot at dim=1
    return one_hot


def one_hot(labels):
    """
     one_hot encoding
    """
    labels = labels.data.cpu().numpy()
    one_hot_encoding = np.zeros((labels.shape[0], num_classes, labels.shape[1], labels.shape[2], labels.shape[3]), \
                       dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot_encoding[:, class_id, ...] = (labels == class_id)
    return one_hot_encoding

def image_show(name, image, resize=5):
    """Show a image using cv2.imshow"""
    H, W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize * W), round(resize * H))


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 3D bilinear kernel suitable for upsampling"""
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size),
                      dtype=np.float64)
    f = math.ceil(kernel_size / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                weight[0, 0, i, j, k] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c)) * (1 - math.fabs(k / f - c))
    for c in range(1, in_channels):
        weight[c, 0, :, :, :] = weight[0, 0, :, :, :]
    return weight


def fill_up_weights(up):
    """
      initial layer weights
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            for k in range(w.size(4)):
                w[0, 0, i, j, k] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c)) * (1 - math.fabs(k / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :, :] = w[0, 0, :, :, :]


def lr_poly(base_lr, iteration, max_iter, power):
    """Learning rate is carried out by polynomial error"""
    return base_lr * ((1 - iteration * 1.0 / max_iter) ** power)
