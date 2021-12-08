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
"""process dataset"""
import numpy as np
from PIL import Image, ImageOps

def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
    """locate crop location"""
    w_step = (image_w - crop_w) // 4
    h_step = (image_h - crop_h) // 4

    ret = list()
    ret.append((0, 0))  # upper left
    ret.append((4 * w_step, 0))  # upper right
    ret.append((0, 4 * h_step))  # lower left
    ret.append((4 * w_step, 4 * h_step))  # lower right
    ret.append((2 * w_step, 2 * h_step))  # center

    if more_fix_crop:
        ret.append((0, 2 * h_step))  # center left
        ret.append((4 * w_step, 2 * h_step))  # center right
        ret.append((2 * w_step, 4 * h_step))  # lower center
        ret.append((2 * w_step, 0 * h_step))  # upper center

        ret.append((1 * w_step, 1 * h_step))  # upper left quarter
        ret.append((3 * w_step, 1 * h_step))  # upper right quarter
        ret.append((1 * w_step, 3 * h_step))  # lower left quarter
        ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

    return ret


class GroupCenterCrop:
    """GroupCenterCrop"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        images = []
        for img in img_group:
            width, height = img.size
            left = (width - self.size)/2
            top = (height - self.size)/2
            right = (width + self.size)/2
            bottom = (height + self.size)/2

            images.append(img.crop((left, top, right, bottom)))
        return images


class GroupNormalize:
    """GroupNormalize"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.shape[0]//len(self.mean))
        rep_std = self.std * (tensor.shape[0]//len(self.std))

        # TODO: make efficient
        for i, _ in enumerate(tensor):
            tensor[i] = (tensor[i]-rep_mean[i])/rep_std[i]
        return tensor


class GroupScale:
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        images = []
        for img in img_group:
            w, h = img.size
            if w > h:
                s = (int(self.size * w / h), self.size)
            else:
                s = (self.size, int(self.size * h / w))
            images.append(img.resize(s, self.interpolation))
        return images


class GroupOverSample:
    """GroupOverSample"""
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        #print(offsets)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class Stack:
    """Stack"""
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        output = []
        if img_group[0].mode == 'L':
            output = np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                output = np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                output = np.concatenate(img_group, axis=2)
        return output


class ToTorchFormatTensor:
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div_sign = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array

            pic = np.array(pic, np.float32)
            pic = np.ascontiguousarray(pic)
            img = pic.transpose((2, 0, 1))
        else:
            # handle PIL Image
            pic = np.array(pic, np.float32)
            pic = np.ascontiguousarray(pic)
            img = pic.reshape(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose((2, 0, 1))

        return img/255. if self.div_sign else img
