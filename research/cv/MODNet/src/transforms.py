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
import random
import copy
import cv2
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion


def create_operators(op_param_list):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """

    assert isinstance(op_param_list, list), 'operator config should be a list'
    config = copy.deepcopy(op_param_list)
    ops = []
    for operator in config:
        assert isinstance(operator, dict), "yaml format error"
        op_name = operator.pop('type')

        if op_name == 'LoadImages':
            ops.append(LoadImages(**operator))
        elif op_name == 'ResizeByShort':
            ops.append(ResizeByShort(**operator))
        elif op_name == 'RandomCrop':
            ops.append(RandomCrop(**operator))
        elif op_name == 'GenTrimap':
            ops.append(GenTrimap(**operator))
        elif op_name == 'RandomHorizontalFlip':
            ops.append(RandomHorizontalFlip(**operator))
        elif op_name == 'Normalize':
            ops.append(Normalize(**operator))
        elif op_name == 'Transpose':
            ops.append(Transpose(**operator))
    return ops

class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].
    """

    def __init__(self, transforms, trans_to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.trans_to_rgb = trans_to_rgb

    def __call__(self, data):
        """
        Args:
            data (dict): The data to transform.

        Returns:
            dict: the transformed data
        """
        for op in self.transforms:
            data = op(data)
            if data is None:
                return None
        return data

class Transpose:
    def __init__(self, trans_list=('image',), unsqueeze_list=('alpha', 'trimap')):
        self.trans_list = trans_list
        self.unsqueeze_list = unsqueeze_list

    def __call__(self, data):
        for key in self.trans_list:
            data[key] = np.transpose(data[key], (2, 0, 1))

        for key in self.unsqueeze_list:
            if key in data.keys():
                data[key] = data[key][None, :, :]

        return data

def resize_short(image, short_size=224, interpolation=cv2.INTER_LINEAR):
    value = min(image.shape[0], image.shape[1])
    scale = float(short_size) / float(value)
    resized_width = int(round(image.shape[1] * scale))
    resized_height = int(round(image.shape[0] * scale))

    image = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation)
    return image


class LoadImages:
    def __init__(self, gt_field=('alpha',), trans_to_rgb=True):
        self.trans_to_rgb = trans_to_rgb
        self.gt_field = gt_field

    def __call__(self, data):
        # print(data['image'])
        if isinstance(data['image'], str):
            data['image'] = cv2.imread(data['image'])
        for key in self.gt_field:
            if isinstance(data[key], str):
                data[key] = cv2.imread(data[key], 0)

        if self.trans_to_rgb:
            data['image'] = cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB)
            for key in data.get('gt_field', []):
                if len(data[key].shape) == 2:
                    continue
                data[key] = cv2.cvtColor(data[key], cv2.COLOR_BGR2RGB)
        return data

class Resize:
    def __init__(self, gt_field=('alpha',), target_size=(512, 512), interp='area'):
        if isinstance(target_size, list or tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.gt_field = gt_field
        self.target_size = target_size
        if interp == 'area':
            self.interps = cv2.INTER_AREA
        else:
            self.interps = cv2.INTER_LINEAR

    def __call__(self, data):
        data['image'] = cv2.resize(data['image'], self.target_size, self.interps)
        for key in self.gt_field:
            if key == 'trimap':
                data[key] = cv2.resize(data[key], self.target_size, cv2.INTER_NEAREST)
            else:
                data[key] = cv2.resize(data[key], self.target_size, self.interps)
        return data

class ResizeByShort:
    def __init__(self, gt_field=('alpha',), ref_size=512, interp='area'):
        if not isinstance(ref_size, int):
            raise TypeError(
                "Type of `ref_size` is invalid. It should be int, but it is {}"
                .format(type(ref_size)))

        self.gt_field = gt_field
        self.ref_size = ref_size
        if interp == 'area':
            self.interps = cv2.INTER_AREA
        else:
            self.interps = cv2.INTER_LINEAR

    def __call__(self, data):
        image = data['image']
        im_h, im_w, _ = image.shape

        if max(im_h, im_w) < self.ref_size or min(im_h, im_w) != self.ref_size:
            if im_w >= im_h:
                im_rh = self.ref_size
                im_rw = int(im_w / im_h * self.ref_size)

            elif im_w < im_h:
                im_rw = self.ref_size
                im_rh = int(im_h / im_w * self.ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        # resize to int mult
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        data['image'] = cv2.resize(data['image'], (im_rw, im_rh), self.interps)
        for key in self.gt_field:
            if key == 'trimap':
                data[key] = cv2.resize(data[key], (im_rw, im_rh), cv2.INTER_NEAREST)
            else:
                data[key] = cv2.resize(data[key], (im_rw, im_rh), self.interps)
        return data

class RandomCrop:
    """
    Randomly crop

    Args:
    crop_size (tuple|list): The size you want to crop from image.
    """

    def __init__(self, gt_field=('alpha',), crop_size=((512, 512), (768, 768), (1024, 1024))):
        if not isinstance(crop_size[0], (list, tuple)):
            crop_size = [crop_size]
        self.crop_size = crop_size
        self.gt_field = gt_field

    def __call__(self, data):
        idex = np.random.randint(low=0, high=len(self.crop_size))
        crop_w, crop_h = self.crop_size[idex]
        img_h, img_w = data['image'].shape[0:2]

        start_h = 0
        start_w = 0
        if img_h > crop_h:
            start_h = np.random.randint(img_h - crop_h + 1)
        if img_w > crop_w:
            start_w = np.random.randint(img_w - crop_w + 1)

        end_h = min(img_h, start_h + crop_h)
        end_w = min(img_w, start_w + crop_w)

        data['image'] = data['image'][start_h:end_h, start_w:end_w]
        for key in self.gt_field:
            data[key] = data[key][start_h:end_h, start_w:end_w]

        return data

class RandomHorizontalFlip:
    """
    Random flip an image.

    Args:
        probability (float, optional): A probability of flipping. Default: 0.5.
    """

    def __init__(self, gt_field=('alpha',), probability=0.5):
        self.probability = probability
        self.gt_field = gt_field

    def __call__(self, data):
        if random.random() < self.probability:
            data['image'] = cv2.flip(data['image'], 1)
            for key in self.gt_field:
                data[key] = cv2.flip(data[key], 1)

        return data

    def horizontal_flip(self, im):
        if len(im.shape) == 3:
            im = im[:, ::-1, :]
        elif len(im.shape) == 2:
            im = im[:, ::-1]
        return im

class GenTrimap:
    def __init__(self, gt_field=('alpha',), alpha_size=512, max_value=1):
        self.gt_field = gt_field
        self.alpha_size = alpha_size
        if max_value == 255:
            self.gen_fn = self.gen_trimap_with_dilate
        else:
            self.gen_fn = self.gen_trimap_sci

    def __call__(self, data):
        alpha = data['alpha']
        data['trimap'] = self.gen_fn(alpha)
        return data

    def gen_trimap_sci(self, alpha):
        trimap = (alpha >= 0.9).astype('float32')
        not_bg = (alpha > 0).astype('float32')

        d_size = self.alpha_size // 256 * 15
        e_size = self.alpha_size // 256 * 15

        trimap[np.where((grey_dilation(not_bg, size=(d_size, d_size))
                         - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
        return trimap

    def gen_trimap_with_dilate(self, alpha):
        kernel_size = random.randint(15, 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
        erode = cv2.erode(fg, kernel, iterations=1)
        trimap = erode *255 + (dilate-erode)*128
        return trimap.astype(np.uint8)

class Normalize:
    """
    Normalize an image.

    Args:
        gt_field (tulip, optional)
        mean (list, optional): The mean value of an image. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of an image. Default: [0.5, 0.5, 0.5].

    """

    def __init__(self, gt_field=('alpha',), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        self.gt_field = gt_field

        if not (isinstance(self.mean,
                           (list, tuple)) and isinstance(self.std,
                                                         (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, data):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        data['image'] = self.normalize(data['image'], mean, std)
        if 'fg' in data.keys():
            data['fg'] = self.normalize(data['fg'], mean, std)
        for key in self.gt_field:
            data[key] = data[key].astype(np.float32, copy=False) / 255.0

        return data

    def normalize(self, im, mean, std):
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        return im
