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
"""definition of some data loading operations"""

from __future__ import print_function, division
import random
import os
import glob

import numpy as np
from skimage import io, transform, color

from mindspore import context
from mindspore import dataset as ds
from mindspore.common import dtype as mstype
import mindspore.dataset.transforms as CC
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size


# ==========================dataset load==========================
class RescaleT:
    """Rescale operation"""

    def __init__(self, output_size):
        """RescaleT definition"""
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """RescaleT compute"""
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
                               preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class Rescale:
    """Rescale operation"""

    def __init__(self, output_size):
        """Rescale definition"""
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """Rescale compute"""
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        a = int(random.random() * 4) % 4
        image = np.rot90(image, a)
        label = np.rot90(label, a)
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class RandomCrop:
    """RandomCrop operation"""

    def __init__(self, output_size):
        """RandomCrop definition"""
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """RandomCrop compute"""
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        a = int(random.random() * 4) % 4
        image = np.rot90(image, a)
        label = np.rot90(label, a)
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """Convert operation"""

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        image = image / np.max(image)
        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return imidx, tmpImg, tmpLbl


class ToTensorLab:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        """ToTensorLab definition"""
        self.flag = flag

    def __call__(self, sample):
        """ToTensorLab compute"""
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the rgb to brg
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx': imidx, 'image': tmpImg, 'label': tmpLbl}


class SalObjDataset:
    """Preprocess the tensor"""

    def __init__(self, img_name_list, lbl_name_list, train_transform=None):
        """
        SalObjDataset definition
        """
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.train_transform = train_transform
        self.rescale = RescaleT(320)
        self.randomcrop = RandomCrop(288)
        self.totensor = ToTensorLab(flag=0)

    def __len__(self):
        """
        get the length of the dataset
        """
        return len(self.image_name_list)

    def __getitem__(self, idx):
        """
        get data in one step
        """
        image = io.imread(self.image_name_list[idx])
        imidx = np.array([idx])

        if not self.label_name_list:
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if len(label_3.shape) == 3:
            label = label_3[:, :, 0]
        elif len(label_3.shape) == 2:
            label = label_3

        if len(image.shape) == 3 and len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        elif len(image.shape) == 2 and len(label.shape) == 2:
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.train_transform:
            sample = self.rescale(sample)
            sample = self.randomcrop(sample)
            sample = self.totensor(sample)
        else:
            sample = self.rescale(sample)
            sample = self.totensor(sample)

        return sample['image'], sample['label']


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


def create_dataset(image_dir, label_dir, args):
    """
    create dataset
    """
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    tra_image_dir = image_dir
    tra_label_dir = label_dir
    image_ext = '.jpg'
    label_ext = '.png'
    tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)
    print("---")
    print(image_dir)
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        train_transform=True)
    if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
        device_num, rank_id = _get_rank_info()
        type_cast = CC.TypeCast(mstype.float32)
        data_set = ds.GeneratorDataset(salobj_dataset, ["image", "label"], num_parallel_workers=8, shuffle=True,
                                       num_shards=device_num, shard_id=rank_id)
        data_set = data_set.map(operations=type_cast, input_columns=["image"], num_parallel_workers=8)
        data_set = data_set.map(operations=type_cast, input_columns=["label"], num_parallel_workers=8)
        data_set = data_set.batch(args.batch_size, drop_remainder=True)
    else:
        type_cast = CC.TypeCast(mstype.float32)
        data_set = ds.GeneratorDataset(salobj_dataset, ["image", "label"], num_parallel_workers=8, shuffle=True)
        data_set = data_set.map(operations=type_cast, input_columns=["image"], num_parallel_workers=8)
        data_set = data_set.map(operations=type_cast, input_columns=["label"], num_parallel_workers=8)
        data_set = data_set.batch(args.batch_size, drop_remainder=True)
    return data_set
