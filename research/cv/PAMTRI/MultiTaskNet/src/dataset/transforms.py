# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""transforms"""
import sys
import random
import collections
import cv2

import mindspore.dataset.vision as vision

if sys.version_info < (3, 3):
    Iterable = collections.Iterable
else:
    Iterable = collections.abc.Iterable

class Compose_Keypt():
    """
    Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, vkeypt):
        for t in self.transforms:
            img = t(img, vkeypt)
        return img

class ToTensor_Keypt():
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """
    def __init__(self):
        self.to_tensor = vision.ToTensor()

    def __call__(self, img, vkeypt):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            vkeypot (36x3 list): 2D keypioints and confidence scores.
        """
        if not len(vkeypt) == 108:
            raise TypeError('vkeypt should have size 108. Got inappropriate size: {}'.format(len(vkeypt)))

        return self.to_tensor(img)

class Normalize_Keypt():
    """
    Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, vkeypt):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            vkeypot (36x3 list): 2D keypioints and confidence scores.
        """
        if not len(vkeypt) == 108:
            raise TypeError('vkeypt should have size 108. Got inappropriate size: {}'.format(len(vkeypt)))

        assert len(self.mean) == len(self.std)
        channels_orig = len(self.mean)
        # channels_new = list(tensor.size())[0]

        channels_new = len(tensor)

        if channels_new > channels_orig:
            mean_avg = sum(self.mean) / float(channels_orig)
            std_avg = sum(self.std) / float(channels_orig)
            self.mean.extend([mean_avg] * (channels_new - channels_orig))
            self.std.extend([std_avg] * (channels_new - channels_orig))

        normalize = vision.Normalize(self.mean, self.std, is_hwc=False)
        return normalize(tensor)

class Resize_Keypt():
    """
    Resize the input image to the given size.

    Args:
        size (sequence): Desired output size (w, h).
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, Iterable) and len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, vkeypt):
        """
        Args:
            img (NumPy array): Image to be scaled.
            vkeypot (36x3 list): 2D keypioints and confidence scores.
        """
        if not len(vkeypt) == 108:
            raise TypeError('vkeypt should have size 108. Got inappropriate size: {}'.format(len(vkeypt)))

        height, width, _ = img.shape

        width_scale = float(self.size[0]) / float(width)
        height_scale = float(self.size[1]) / float(height)
        for k in range(len(vkeypt)):
            if k % 3 == 0:
                vkeypt[k] *= width_scale
            elif k % 3 == 1:
                vkeypt[k] *= height_scale

        return cv2.resize(img, dsize=self.size, interpolation=self.interpolation)

class RandomHorizontalFlip_Keypt():
    """
    Horizontally flip the given image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, vkeypt):
        """
        Args:
            img (NumPy array): Image to be flipped.
            vkeypot (36x3 list): 2D keypioints and confidence scores.
        """
        if random.random() < self.p:
            if not len(vkeypt) == 108:
                raise TypeError('vkeypt should have size 108. Got inappropriate size: {}'.format(len(vkeypt)))

            _, width, _ = img.shape
            for k in range(len(vkeypt)):
                if k % 3 == 0:
                    vkeypt[k] = width - vkeypt[k] - 1

            img = cv2.flip(img, flipCode=1)

        return img

class Random2DTranslation_Keypt():
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        size (sequence): Desired output size (w, h).
        p (float): probability of performing this transformation. Default: 0.5.
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """
    def __init__(self, size, p=0.5, interpolation=cv2.INTER_LINEAR):
        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img, vkeypt):
        """
        Args:
            img (NumPy array): Image to be cropped.
            vkeypot (36x3 list): 2D keypioints and confidence scores.
        """
        if not len(vkeypt) == 108:
            raise TypeError('vkeypt should have size 108. Got inappropriate size: {}'.format(len(vkeypt)))

        height, width, _ = img.shape

        if random.uniform(0, 1) > self.p:
            width_scale = float(self.size[0]) / float(width)
            height_scale = float(self.size[1]) / float(height)
            for k in range(len(vkeypt)):
                if k % 3 == 0:
                    vkeypt[k] *= width_scale
                elif k % 3 == 1:
                    vkeypt[k] *= height_scale

            return cv2.resize(img, dsize=self.size, interpolation=self.interpolation)

        width_new, height_new = int(round(self.size[0] * 1.125)), int(round(self.size[1] * 1.125))

        width_scale = float(width_new) / float(width)
        height_scale = float(height_new) / float(height)
        for k in range(len(vkeypt)):
            if k % 3 == 0:
                vkeypt[k] *= width_scale
            elif k % 3 == 1:
                vkeypt[k] *= height_scale

        img_resized = cv2.resize(img, dsize=(width_new, height_new), interpolation=self.interpolation)
        x_maxrange = width_new - self.size[0]
        y_maxrange = height_new - self.size[1]
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))

        for k in range(len(vkeypt)):
            if k % 3 == 0:
                vkeypt[k] -= x1
            elif k % 3 == 1:
                vkeypt[k] -= y1

        return img_resized[y1 : (y1 + self.size[1]), x1 : (x1 + self.size[0])]
