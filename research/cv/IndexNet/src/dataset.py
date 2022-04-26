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
"""Dataset scripts."""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import morphology


def resize_image_alpha(image, alpha, nh, nw):
    """
    Resize image and alpha concatenated input to the same size.

    Args:
        image (np.array): Input 4 channels image.
        alpha (np.array): Stacked alpha, mask, fg, bg and image.
        nh (int): Height resize size.
        nw (int): Width resize size.

    Returns:
        image (np.array): Resized input 4 channels image.
        alpha (np.array): Resized stacked alpha, mask, fg, bg and image.
    """
    alpha_chn = alpha.shape[2]

    trimap = image[:, :, 3]
    image = image[:, :, 0:3]
    mask = alpha[:, :, 1]
    if alpha_chn > 2:
        fg = alpha[:, :, 2:5]
        bg = alpha[:, :, 5:8]
        ori_image = alpha[:, :, 8:11]
    alpha = alpha[:, :, 0]

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    trimap = cv2.resize(trimap, (nw, nh), interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if alpha_chn > 2:
        fg = cv2.resize(fg, (nw, nh), interpolation=cv2.INTER_CUBIC)
        bg = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_CUBIC)
        ori_image = cv2.resize(ori_image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    trimap = trimap.reshape(trimap.shape[0], trimap.shape[1], 1)
    alpha = alpha.reshape(alpha.shape[0], alpha.shape[1], 1)
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

    image = np.concatenate((image, trimap), axis=2)
    if alpha_chn > 2:
        alpha = np.concatenate((alpha, mask, fg, bg, ori_image), axis=2)
    else:
        alpha = np.concatenate((alpha, mask), axis=2)

    return image, alpha


class RandomCrop:
    """
    Randomly crop the image.

    Args:
        output_size (int): Desired output size. If int, square crop is made.
        scales (list): Desired scales
    """
    def __init__(self, output_size, scales):
        assert isinstance(output_size, int)

        self.output_size = output_size
        self.scales = scales

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        h, w = image.shape[:2]

        if min(h, w) < self.output_size:
            s = (self.output_size + 180) / min(h, w)
            nh, nw = int(np.floor(h * s)), int(np.floor(w * s))
            image, alpha = resize_image_alpha(image, alpha, nh, nw)
            h, w = image.shape[:2]

        crop_size = np.floor(self.output_size * np.array(self.scales)).astype('int')
        crop_size = crop_size[crop_size < min(h, w)]
        crop_size = int(random.choice(crop_size))

        c = int(np.ceil(crop_size / 2))
        mask = np.equal(image[:, :, 3], 128).astype(np.uint8)
        if mask[c:h - c + 1, c:w - c + 1].sum() != 0:
            mask_center = np.zeros((h, w), dtype=np.uint8)
            mask_center[c:h - c + 1, c:w - c + 1] = 1
            mask = (mask & mask_center)
            idh, idw = np.where(mask == 1)
            ids = random.choice(range(len(idh)))
            hc, wc = idh[ids], idw[ids]
            h1, w1 = hc - c, wc - c
        else:
            idh, idw = np.where(mask == 1)
            ids = random.choice(range(len(idh)))
            hc, wc = idh[ids], idw[ids]
            h1, w1 = np.clip(hc - c, 0, h), np.clip(wc - c, 0, w)
            h2, w2 = h1 + crop_size, w1 + crop_size
            h1 = h - crop_size if h2 > h else h1
            w1 = w - crop_size if w2 > w else w1

        image = image[h1:h1 + crop_size, w1:w1 + crop_size, :]
        alpha = alpha[h1:h1 + crop_size, w1:w1 + crop_size, :]

        if crop_size != self.output_size:
            nh = nw = self.output_size
            image, alpha = resize_image_alpha(image, alpha, nh, nw)

        return {'image': image, 'alpha': alpha}


class RandomFlip:
    """
    Randomly flip the image and alpha.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            alpha = cv2.flip(alpha, 1)
        return {'image': image, 'alpha': alpha}


class Transpose:
    """
    Transpose arrays to the input view.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        # swap color axis
        # from numpy image: H x W x C
        # to numpy image: C X H X W
        image = image.transpose((2, 0, 1))
        alpha = alpha.transpose((2, 0, 1))
        return {'image': image, 'alpha': alpha}


class ResizePad:
    """
    Resize pad and transpose input image (if necessary).
    If image is vertical, transpose, resize, pad.

    Args:
        size (tuple): Input image size.
    """
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        h, w = image.shape[:2]

        transp = False
        # Flip image to horizontal state
        if w < h:
            transp = True
            image = image.transpose(1, 0, 2)
            alpha = alpha.transpose(1, 0, 2)
            h, w = w, h

        if self.w < w or self.h < h:
            aspect_ratio = min(self.h / h, self.w / w)
        else:
            aspect_ratio = max(w / self.w, h / self.h)

        nh, nw = int(h * aspect_ratio), int(w * aspect_ratio)

        image, alpha = resize_image_alpha(image, alpha, nh, nw)

        image_mask = np.ones_like(image[:, :, 0])
        image = np.pad(image, ((0, self.h - nh), (0, self.w - nw), (0, 0)))
        image_mask = np.pad(image_mask, ((0, self.h - nh), (0, self.w - nw))).astype(np.bool)

        return {'image': image, 'alpha': alpha}, transp, image_mask, (nh, nw)


class Normalize:
    """
    Normalize image.

    Args:
        scale (float): Scale for image.
        mean (np.array): 4 dims mean to normalize every axis.
        std (np.array): 4 dims std to normalize every axis.
    """
    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        image, alpha = image.astype('float32'), alpha.astype('float32')

        image = (self.scale * image - self.mean) / self.std
        alpha[:, :, 0] = self.scale * alpha[:, :, 0]
        if alpha.shape[2] > 2:
            alpha[:, :, 2:11] = self.scale * alpha[:, :, 2:11]
        return {'image': image.astype('float32'), 'alpha': alpha.astype('float32')}


class BaseDataset:
    """
    Base dataset class for Image Matting.

    Args:
        data_dir (str): Path to the dataset folder.
        config: Config parameters.
        sub_folder (str): Name of sub folder to train/test part of dataset.
        data_file (str): Name of data schema.
    """
    def __init__(self, data_dir, config, sub_folder='train', data_file='data.txt'):
        self.data_dir = Path(data_dir)

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f'Not valid path to merged dataset sub folder: {Path(data_dir, sub_folder)}')

        with Path(data_dir, sub_folder, data_file).open() as file:
            self.datalist = [name.split('|') for name in file.read().splitlines()]

        self.scales = config.scales
        self.img_scale = 1. / float(config.img_scale)
        self.img_mean = np.array(config.img_mean).reshape((1, 1, 4))
        self.img_std = np.array(config.img_std).reshape((1, 1, 4))

    def __len__(self):
        return len(self.datalist)

    @staticmethod
    def _read_image(image_path):
        """
        Read image.

        Args:
            image_path: Image path.

        Returns:
            img_arr (np.array): Loaded image.
        """
        img_arr = np.array(Image.open(image_path))
        if len(img_arr.shape) == 2:  # grayscale
            img_arr = np.tile(img_arr, [3, 1, 1]).transpose((1, 2, 0))
        return img_arr


class ImageMattingDatasetTrain(BaseDataset):
    """
    Image Matting train dataset.

    Args:
        data_dir (str): Path to the dataset folder.
        bg_dir (str): Path to the backgrounds dataset folder.
        config: Config parameters.
        sub_folder (str): Name of sub folder to train part of dataset.
        data_file (str): Name of data schema.
    """
    def __init__(self, data_dir, bg_dir, config, sub_folder, data_file):
        super().__init__(data_dir, config, sub_folder, data_file)
        self.bg_dir = Path(bg_dir)
        self.crop_size = config.input_size

        if not self.bg_dir.is_dir():
            raise NotADirectoryError(f'Not valid path to backgrounds: {self.bg_dir}')

    @staticmethod
    def _generate_trimap(alpha):
        """
        Generate trimap with random line width.

        Args:
            alpha (np.array): Image alpha channel (mask).

        Returns:
            trimap (np.array): Generated trimap.
        """
        # alpha \in [0, 1] should be taken into account
        # be careful when dealing with regions of alpha=0 and alpha=1
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
        unknown = unknown - fg
        # image dilation implemented by Euclidean distance transform
        unknown = morphology.distance_transform_edt(unknown == 0) <= np.random.randint(1, 20)
        trimap = fg * 255
        trimap[unknown] = 128
        return trimap.astype(np.uint8)

    def __getitem__(self, idx):
        image_name = self.data_dir / self.datalist[idx][0]
        alpha_name = self.data_dir / self.datalist[idx][1]
        fg_name = self.data_dir / self.datalist[idx][2]
        bg_name = self.bg_dir / self.datalist[idx][3]

        image = self._read_image(image_name)
        fg = self._read_image(fg_name)
        bg = self._read_image(bg_name)

        alpha = np.array(Image.open(alpha_name))
        if alpha.ndim != 2:
            alpha = alpha[:, :, 0]

        fgh, fgw = fg.shape[0:2]
        bgh, bgw = bg.shape[0:2]
        rh, rw = fgh / float(bgh), fgw / float(bgw)
        r = rh if rh > rw else rw
        nh, nw = int(np.ceil(bgh * r)), int(np.ceil(bgw * r))
        bg = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_CUBIC)
        bg = bg[0:fgh, 0:fgw, :]

        trimap = self._generate_trimap(alpha)
        mask = np.equal(trimap, 128).astype(np.uint8)

        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        trimap = trimap.reshape((trimap.shape[0], trimap.shape[1], 1))
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

        alpha = np.concatenate((alpha, mask, fg, bg, image), axis=2)
        image = np.concatenate((image, trimap), axis=2)

        sample = {'image': image, 'alpha': alpha}

        # Apply training transforms
        sample = RandomCrop(self.crop_size, self.scales)(sample)
        sample = RandomFlip()(sample)
        sample = Normalize(self.img_scale, self.img_mean, self.img_std)(sample)
        sample = Transpose()(sample)

        image = sample['image']
        alpha = sample['alpha'][0, :, :]
        mask = sample['alpha'][1, :, :]
        fg = sample['alpha'][2:5, :, :]
        bg = sample['alpha'][5:8, :, :]
        c_g = sample['alpha'][8:11, :, :]

        return image, mask, alpha, fg, bg, c_g


class ImageMattingDatasetVal(BaseDataset):
    """
    Image Matting validation dataset.

    Args:
        data_dir (str): Path to the dataset folder.
        config: Config parameters.
        sub_folder (str): Name of sub folder to test part of dataset.
        data_file (str): Name of data schema.
    """
    def __init__(self, data_dir, config, sub_folder, data_file):
        super().__init__(data_dir, config, sub_folder, data_file)
        self.img_size = config.img_size

    def __getitem__(self, idx):
        image_name = self.data_dir / self.datalist[idx][0]
        alpha_name = self.data_dir / self.datalist[idx][1]
        trimap_name = self.data_dir / self.datalist[idx][2]

        image = self._read_image(image_name)
        alpha = np.array(Image.open(alpha_name))
        if alpha.ndim != 2:
            alpha = alpha[:, :, 0]

        trimap = np.array(Image.open(trimap_name))

        alpha = alpha[:, :, 0] if alpha.ndim == 3 else alpha
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        trimap = trimap.reshape((trimap.shape[0], trimap.shape[1], 1))

        image = np.concatenate((image, trimap), axis=2)
        alpha = np.concatenate((alpha, trimap), axis=2)

        sample = {'image': image, 'alpha': alpha}

        # Apply validation transforms
        sample, transposed, pad_mask, clear_size = ResizePad(self.img_size)(sample)
        sample = Normalize(self.img_scale, self.img_mean, self.img_std)(sample)
        sample = Transpose()(sample)

        image = sample['image']
        alpha = sample['alpha'][0, :, :]
        mask = sample['alpha'][1, :, :]

        output = image, alpha, mask, transposed, pad_mask, clear_size

        return output
