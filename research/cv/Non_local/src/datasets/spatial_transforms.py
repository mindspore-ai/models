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

# This file was copied from project [feiyunzhang][i3d-non-local-pytorch]

import random
import numbers
from PIL import Image, ImageOps
import numpy as np
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision


class GroupRandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size
        out_images = list()
        x1 = random.randint(0, h - th)
        y1 = random.randint(0, w - tw)
        crop_op = c_vision.Crop((x1, y1), (th, tw))
        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(crop_op(img))

        return out_images


class GroupCornerCrop:

    def __init__(self, size, crop_position=None):
        self.size = (int(size), int(size))
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']
        self.crop_position = crop_position
        if crop_position is None:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]
        else:
            self.randomize = False

    def __call__(self, img_group, crop_position_index):
        w, h = img_group[0].size
        th, tw = self.size
        x1 = 0
        y1 = 0
        out_images = list()
        if self.crop_positions[crop_position_index] == 'c':
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
        elif self.crop_positions[crop_position_index] == 'tl':
            x1 = 0
            y1 = 0
        elif self.crop_positions[crop_position_index] == 'tr':
            x1 = w - tw
            y1 = 0

        crop_op = c_vision.Crop((y1, x1), (th, tw))
        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(crop_op(img))

        return out_images


class GroupCenterCrop:
    def __init__(self, size):
        self.worker = py_vision.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.worker = c_vision.RandomColorAdjust(brightness, contrast, saturation, hue)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip:
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        horizontalflip = c_vision.HorizontalFlip()
        if v < 0.5:
            ret = [horizontalflip(img) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        return img_group


class GroupNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.shape[0] // len(self.mean))
        rep_std = self.std * (tensor.shape[0] // len(self.std))

        # TODO: make efficient
        for i in range(len(self.mean)):
            tensor[i] = (tensor[i] - rep_mean[i]) / rep_std[i]
        return tensor


class GroupScale:
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size):
        self.worker = py_vision.Resize(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomResizeCrop:
    """
    random resize image to shorter size = [256,320] (e.g.),
    and random crop image to 224[e.g.]
    p.s.: if input size > 224, resize_range should be enlarged in equal proportion
    """

    def __init__(self, resize_range, input_size, interpolation=Image.BILINEAR):
        self.resize_range = resize_range
        self.crop_worker = GroupRandomCrop(input_size)
        self.interpolation = interpolation

    def __call__(self, img_group):
        resize_size = random.randint(self.resize_range[0], self.resize_range[1])
        resize_worker = GroupScale(resize_size)
        resized_img_group = resize_worker(img_group)
        crop_img_group = self.crop_worker(resized_img_group)

        return crop_img_group


class Stack:

    def __call__(self, img_group):
        stacked_group = np.concatenate([np.expand_dims(x, 3) for x in img_group], axis=3)

        return stacked_group


class ToTorchFormatTensor:
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C x D) in the range [0, 255]
    to a torch.FloatTensor of shape (C x D x H x W) in the range [0.0, 1.0] """

    def __call__(self, pic):
        # handle numpy arraya
        img = pic.transpose(2, 3, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255
        return img


class SpatialTransformer:

    def __init__(self, resize_range, input_size, brightness, contrast, saturation, hue, mean, std):
        self.resize_range = resize_range
        self.input_size = input_size
        self.colorjitter = c_vision.RandomColorAdjust(brightness, contrast, saturation, hue)
        self.mean = mean
        self.std = std

    def __call__(self, img_group):
        resize_size = random.randint(self.resize_range[0], self.resize_range[1])
        resize_worker = c_vision.Resize(resize_size)
        th, tw = (int(self.input_size), int(self.input_size))
        w, h = resize_worker(img_group[0]).shape[0:2]
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        crop_op = c_vision.Crop((x1, y1), (th, tw))
        v = random.random()
        horizontalflip = c_vision.HorizontalFlip()
        out = []
        for img in img_group:
            img = resize_worker(img)
            assert (img.shape[0] == w and img.shape[1] == h)
            if w == tw and h == th:
                pass
            else:
                img = crop_op(img)
            if v < 0.5:
                img = horizontalflip(img)
            img = self.colorjitter(img)
            img = img.astype(np.float32) / 255
            out.append(img)
        out = np.stack(out, 0).transpose(3, 0, 1, 2)
        return out


class IdentityTransform:

    def __call__(self, data):
        return data
