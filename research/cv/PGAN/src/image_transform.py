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
"""image transform"""
import os

import numpy as np
from PIL import Image


class NumpyResize():
    """NumpyResize"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        r"""
        Args:
            img (np array): image to be resized
        Returns:
            np.Array: resized image
        """
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        np_image = np.array(img.resize(self.size, resample=Image.BILINEAR))
        return np_image


class Crop():
    """Crop"""
    def __init__(self, cx=89, cy=121):
        self.cx = cx
        self.cy = cy

    def __call__(self, img):
        r"""
        Args:
            img (np array): image to be cropped
        Returns:
            np.Array: cropped image
        """
        return np.array(img)[self.cy - 64: self.cy + 64, self.cx - 64: self.cx + 64]


class Resize():
    """Resize"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): image to be resized
        Returns:
            PIL.Image: resized image
        """
        interpolation = 2
        return img.resize(self.size, interpolation)


class TransporeAndDiv():
    """TransporeAndDiv"""
    def __init__(self):
        return

    def __call__(self, img):
        """
        Args:
            img (np array): image to be Transpored
        Returns:
            np array: Transpored and div image
        """
        return img.transpose(2, 0, 1) / 255


class NpToImage():
    """NpToImage"""
    def __init__(self):
        return

    def __call__(self, img):
        """
        Args:
            img (np array): np array to be Image
        Returns:
            PIL.Image: The Image given the numpy array
        """

        mode = 'RGB'
        return Image.fromarray(img, mode=mode)


class TransporeAndMul():
    """TransporeAndMul"""
    def __init__(self):
        return

    def __call__(self, img):
        """
        Args:
            img (np array): image to be Transpore
        Returns:
            np array: Transpored and mul image
        """
        npimg = (img * 255).transpose(1, 2, 0)
        mode = 'RGB'
        return Image.fromarray(np.uint8(npimg), mode=mode)


class Normalize():
    """Normalize"""
    def __init__(self, means, std):
        self.means = means
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (np array): image to be normalized

        Returns:
            np array: Normalized image
        """
        shape = img.shape
        if shape[0] != len(self.means) or shape[0] != len(self.std):
            raise AttributeError("number of channels should equal with the length of means")
        means = self.padding(self.means, shape)
        std = self.padding(self.std, shape)
        result = (img - means) / std
        return result.astype("float32")

    def padding(self, data, shape):
        result = []
        for c in range(shape[0]):  # shape
            value = data[c]
            img = [[value for i in range(shape[2])] for i in range(shape[1])]
            result.append(img)
        return np.array(result)


def pil_loader(path):
    imgExt = os.path.splitext(path)[1]
    if imgExt == ".npy":
        img = np.load(path)[0]
        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
