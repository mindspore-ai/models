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
# ===========================================================================

"""
    base dataset.
"""
import os
import random
import numpy as np
from PIL import Image
import mindspore.dataset.vision as py_vision
import mindspore.dataset.transforms as py_transforms
from src.utils.config import config

IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP", ".tiff"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir_path):
    images = []
    assert os.path.isdir(dir_path), "%s is not a valid directory" % dir_path

    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def normalize():
    return py_vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False)


def get_params(size):
    """
    Get parameters from images.
    """
    w, h = size
    new_h = h
    new_w = w
    if config.resize_or_crop == "resize_and_crop":
        new_h = new_w = config.load_size
    elif config.resize_or_crop == "scale_width_and_crop":
        new_w = config.load_size
        new_h = config.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - config.fine_size))
    y = random.randint(0, np.maximum(0, new_h - config.fine_size))
    is_flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": is_flip}


def get_transform(params, method=Image.BICUBIC, is_normalize=True, is_train=True):
    transform_list = []
    if "resize" in config.resize_or_crop:
        w_h = (config.load_size, config.load_size)
        transform_list.append(Resize(w_h, method))
    elif "scale_width" in config.resize_or_crop:
        transform_list.append(ScaleWidth(config.load_size, method))

    if "crop" in config.resize_or_crop:
        transform_list.append(Crop(params["crop_pos"], config.fine_size))

    if config.resize_or_crop == "none":
        base = float(2**config.n_downsample_global)
        if config.netG == "local":
            base *= 2**config.n_local_enhancers
        transform_list.append(MakePower2(base, method))

    if is_train and not config.no_flip:
        transform_list.append(Flip(params["flip"]))

    transform_list.append(py_vision.ToTensor())

    if is_normalize:
        transform_list.append(py_vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False))
    return py_transforms.Compose(transform_list)


class MakePower2:
    """
    make power 2
    """

    def __init__(self, base, method):
        self.base = base
        self.method = method

    def __call__(self, img):
        ow, oh = img.size
        h = int(round(oh / self.base) * self.base)
        w = int(round(ow / self.base) * self.base)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), self.method)


class ScaleWidth:
    """scale width"""

    def __init__(self, target_width, method):
        self.target_width = target_width
        self.method = method

    def __call__(self, img):
        ow, oh = img.size
        if ow == self.target_width:
            return img
        w = self.target_width
        h = int(self.target_width * oh / ow)
        return img.resize((w, h), self.method)


class Crop:
    """Crop"""

    def __init__(self, pos, size):
        self.pos = pos
        self.size = size

    def __call__(self, img):
        ow, oh = img.size
        x1, y1 = self.pos
        tw = th = self.size
        if ow > tw or oh > th:
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img


class Flip:
    """flip"""

    def __init__(self, is_flip):
        self.is_flip = is_flip

    def __call__(self, img):
        if self.is_flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class Resize:
    """Resize"""

    def __init__(self, w_h, method):
        self.w_h = w_h
        self.method = method

    def __call__(self, img):
        return img.resize(self.w_h, self.method)
