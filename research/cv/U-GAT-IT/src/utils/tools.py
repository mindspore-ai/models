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
"""tools"""

import os
import cv2
import numpy as np
from scipy import misc

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir

def str2bool(x):
    return x.lower() in 'true'

def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.asnumpy().transpose(1, 2, 0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
