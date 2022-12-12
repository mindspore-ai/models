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
# This file refers to the project https://github.com/MhLiao/DB.git

"""DBNet Dataset DataLoader"""
import os
import math
import glob
import cv2
import numpy as np

from mindspore import dataset as ds
from mindspore.dataset.vision import RandomColorAdjust, ToTensor, ToPIL

from .pre_process import MakeSegDetectionData, MakeBorderMap
from .random_thansform import RandomAugment


def get_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype('float32')
    return img


def scale_pad(img, polys, eval_size):
    """scale image and polys with short side, then pad to eval_size."""
    h, w, c = img.shape
    s_h = eval_size[0] / h
    s_w = eval_size[1] / w
    scale = min(s_h, s_w)
    new_h = int(scale * h)
    new_w = int(scale * w)
    img = cv2.resize(img, (new_w, new_h))
    padimg = np.zeros((eval_size[0], eval_size[1], c), img.dtype)
    padimg[:new_h, :new_w, :] = img
    polys = polys * scale
    return padimg, polys


def get_bboxes(gt_path, config):
    """Get polys and it's `dontcare` flag by gt_path."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    polys = []
    dontcare = []
    for line in lines:
        line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if "#" in gt[-1]:
            dontcare.append(True)
        else:
            dontcare.append(False)
        if config.dataset.is_icdar2015:
            box = [int(gt[i]) for i in range(8)]
        else:
            box = [int(gt[i]) for i in range(len(gt) - 1)]
        polys.append(box)
    return np.array(polys), np.array(dontcare)


def resize(img, polys=None, denominator=32, isTrain=True):
    """Resize image and its polys."""
    w_scale = math.ceil(img.shape[1] / denominator) * denominator / img.shape[1]
    h_scale = math.ceil(img.shape[0] / denominator) * denominator / img.shape[0]
    img = cv2.resize(img, dsize=None, fx=w_scale, fy=h_scale)
    if polys is None:
        return img
    if isTrain:
        new_polys = []
        for poly in polys:
            poly[:, 0] = poly[:, 0] * w_scale
            poly[:, 1] = poly[:, 1] * h_scale
            new_polys.append(poly)
        polys = new_polys
    else:
        polys[:, :, 0] = polys[:, :, 0] * w_scale
        polys[:, :, 1] = polys[:, :, 1] * h_scale
    return img, polys


class IC15DataLoader():
    """IC15 DataLoader"""
    def __init__(self, config, isTrain=True):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.config = config
        self.isTrain = isTrain

        self.ra = RandomAugment(max_tries=config.dataset.random_crop.max_tries,
                                min_crop_side_ratio=config.dataset.random_crop.min_crop_side_ratio)
        self.ms = MakeSegDetectionData(config.train.min_text_size,
                                       config.train.shrink_ratio)
        self.mb = MakeBorderMap(config.train.shrink_ratio,
                                config.train.thresh_min, config.train.thresh_max)

        if isTrain:
            img_paths = glob.glob(os.path.join(config.train.img_dir,
                                               '*' + config.train.img_format))
        else:
            img_paths = glob.glob(os.path.join(config.eval.img_dir,
                                               '*' + config.eval.img_format))

        if self.isTrain:
            img_dir = config.train.gt_dir
            if config.dataset.is_icdar2015:
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.jpg.txt')
                            for img_path in img_paths]
            else:
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.txt')
                            for img_path in img_paths]
        else:
            img_dir = config.eval.gt_dir
            if config.dataset.is_icdar2015:
                gt_paths = [os.path.join(img_dir, 'gt_' + img_path.split('/')[-1].split('.')[0] + '.txt')
                            for img_path in img_paths]
            else:
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.txt')
                            for img_path in img_paths]

        self.img_paths = img_paths
        self.gt_paths = gt_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        # Getting
        img = get_img(img_path)
        original = resize(img)
        polys, dontcare = get_bboxes(gt_path, self.config)

        # Random Augment
        if self.isTrain and self.config.train.is_transform:
            img, polys = self.ra.random_scale(img, polys, self.config.dataset.short_side)
            img, polys = self.ra.random_rotate(img, polys, self.config.dataset.random_angle)
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = polys.reshape((polys.shape[0], polys.shape[1] // 2, 2))
        img, polys = resize(img, polys, isTrain=self.isTrain)

        # Post Process
        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            polys = np.array(polys)
            dontcare = np.array(dontcare, dtype=np.bool8)
            img, polys = scale_pad(img, polys, self.config.eval.eval_size)

        # Show Images
        if self.config.dataset.is_show:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0]*255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask*255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map*255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask*255)

        # Random Colorize
        if self.isTrain and self.config.train.is_transform:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img.astype(np.uint8)))

        # Normalize
        img -= self.RGB_MEAN
        img = ToTensor()(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        return original, img, polys, dontcare


class TotalTextDataLoader():
    def __init__(self, config, isTrain=True):
        if isTrain:
            img_paths = glob.glob(os.path.join(config['train']['img_dir'],
                                               '*' + config['train']['img_format']))
        else:
            img_paths = glob.glob(os.path.join(config['eval']['img_dir'],
                                               '*' + config['eval']['img_format']))
        self.img_paths = img_paths
        if self.isTrain:
            img_dir = config['train']['gt_dir']
            gt_paths = [os.path.join(img_dir, 'poly_gt_' + img_path.split('/')[-1].split('.')[0] + '.mat')
                        for img_path in self.img_paths]

        else:
            img_dir = config['eval']['gt_dir']
            gt_paths = [os.path.join(img_dir, 'poly_gt_' + img_path.split('/')[-1].split('.')[0] + '.mat')
                        for img_path in self.img_paths]

        self.gt_paths = gt_paths

    def get_bboxes(self, gt_path):
        """
        Process a mat, that is, process the polygons of an image
        :param anno_file: mat file path
        """
        from scipy.io import loadmat
        gt_dict = loadmat(gt_path)
        arr = gt_dict['polygt']
        polys = []
        dontcare = []
        for line in range(0, arr.shape[0]):
            x_arr = arr[line][1]
            y_arr = arr[line][3]
            content = arr[line][4]
            if content.shape[0] == 0:
                content_str = '#'
            else:
                content_str = content.item()
            poly = []
            for i in range(0, x_arr.shape[1]):
                poly.append(x_arr[0][i])
                poly.append(y_arr[0][i])
            polys.append(poly)
            if "#" in content_str:
                dontcare.append(True)
            else:
                dontcare.append(False)

        return polys, np.array(dontcare)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        img = get_img(img_path)
        original = resize(img)
        polys, dontcare = self.get_bboxes(gt_path)

        if self.isTrain and self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, self.config['dataset']['short_side'])
            img, polys = self.ra.random_rotate(img, polys, self.config['dataset']['random_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = self.polys_convert(polys)

        img, polys = resize(img, polys, isTrain=self.isTrain)
        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            dontcare = np.array(dontcare, dtype=np.bool8)

        # Show Images
        if self.config['dataset']['is_show']:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0] * 255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask * 255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map * 255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask * 255)

        # Random Colorize
        if self.isTrain and self.config['train']['is_transform']:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img.astype(np.uint8)))

        # Normalize
        img -= self.RGB_MEAN
        img = ToTensor()(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        return original, img, polys, dontcare


def create_dataset(config, is_train):
    """Create MindSpore Dataset object."""
    ds.config.set_prefetch_size(config.dataset.prefetch_size)
    if config.dataset.type == "IC15":
        data_loader = IC15DataLoader(config, isTrain=is_train)
    elif config.dataset.type == "TotalText":
        data_loader = TotalTextDataLoader(config, isTrain=is_train)
    else:
        raise ValueError(f"Not support dataset.type: {config.dataset.type}.")
    if not hasattr(config, "device_num"):
        config.device_num = 1
    if not hasattr(config, "rank_id"):
        config.rank_id = 0
    if is_train:
        dataset = ds.GeneratorDataset(data_loader,
                                      ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'],
                                      num_parallel_workers=config.dataset.num_workers,
                                      num_shards=config.device_num, shard_id=config.rank_id,
                                      shuffle=True, max_rowsize=config.dataset.max_rowsize)
    else:
        dataset = ds.GeneratorDataset(data_loader, ['original', 'img', 'polys', 'dontcare'])
    batch_size = config.train.batch_size if is_train else 1
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    steps_pre_epoch = dataset.get_dataset_size()
    return dataset, steps_pre_epoch
