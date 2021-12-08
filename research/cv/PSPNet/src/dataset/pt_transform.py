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
Functions for input transform
"""
import random
import math
import numbers
import collections
import numpy as np
import cv2


class Compose:
    """ compose the process functions """

    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
        for t in self.segtransform:
            image, label = t(image, label)
        return image, label


class Normalize:
    """
    Normalize tensor with mean and standard deviation along channel:

        channel = (channel - mean) / std

    """

    def __init__(self, mean, std=None, is_train=True):
        if std is None:
            assert mean
        else:
            assert len(mean) == len(std)
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.is_train = is_train

    def __call__(self, image, label):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                               "[eg: data read by cv2.imread()].\n")
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n")
        image = np.transpose(image, (2, 0, 1))  # (473, 473, 3) -> (3, 473, 473)

        if self.is_train:
            if self.std is None:
                image = image - self.mean[:, None, None]
            else:
                image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        return image, label


class Resize:
    """Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w). """

    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, label):
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label


class RandScale:
    """ Randomly resize image & label with scale factor in [scale_min, scale_max] """

    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise RuntimeError("segtransform.RandScale() scale param error.\n")
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise RuntimeError("segtransform.RandScale() aspect_ratio param error.\n")

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop:
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        # [473, 473], 'rand', padding=mean, ignore255
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("crop size error.\n")
        if crop_type in ('center', 'rand'):
            self.crop_type = crop_type
        else:
            raise RuntimeError("crop type error: rand | center\n")
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise RuntimeError("padding in Crop() should be a number list\n")
            if len(padding) != 3:
                raise RuntimeError("padding channel is not equal with 3\n")
        else:
            raise RuntimeError("padding in Crop() should be a number list\n")
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise RuntimeError("ignore_label should be an integer number\n")

    def __call__(self, image, label):
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise RuntimeError("segtransform.Crop() need padding while padding argument is None\n")
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)

        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        return image, label


class RandRotate:
    """
    Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    """

    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise RuntimeError("segtransform.RandRotate() scale param error.\n")
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise RuntimeError("padding in RandRotate() should be a number list\n")
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):

        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip:
    """ Random Horizontal Flip """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip:
    """ Random Vertical Flip """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur:
    """
    RandomGaussianBlur
    """

    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RGB2BGR:
    """
    Converts image from RGB order to BGR order
    """

    def __init__(self):
        pass

    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB:
    """
    Converts image from BGR order to RGB order
    """
    def __init__(self):
        pass

    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label
