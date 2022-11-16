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
"""sen1-2 dataloader"""
import os
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as transform
from mindspore import dtype as mstype
import cv2


class opt_sar_dataset:
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self,
                 root,
                 train=True):
        self.train = train
        self.root = root

        self.train_data_path = os.path.join(self.root, 'train.npz')
        self.test_data_path = os.path.join(self.root, 'test.npz')

        print('loading data---')
        self.train_data = np.load(self.train_data_path)
        self.test_data = np.load(self.test_data_path)

        if self.train:
            self.train_img = self.train_data['arr_0']
            self.train_label = self.train_data['arr_1']
            self.train_len = self.train_img.shape[0]

        else:
            self.test_img = self.test_data['arr_0']
            self.test_label = self.test_data['arr_1']
            self.test_len = self.test_img.shape[0]

        print('loading data done!!!')

    def __getitem__(self, index):
        if self.train:
            img = self.train_img[index]
            label = self.train_label[index]
        else:
            img = self.test_img[index]
            label = self.test_label[index]

        opt_img = img[:, :64]
        sar_img = img[:, 64:]
        return opt_img, sar_img, label

    def __len__(self):
        if self.train:
            return self.train_len
        return self.test_len


class DataAugment:
    def __init__(self,
                 shape,
                 mean_image,
                 std_image):

        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape
        self.mean_image = mean_image
        self.std_image = std_image

    def img_resize(self, img):
        return cv2.resize(img, self.shape, interpolation=cv2.INTER_LINEAR)

    def __call__(self, opt_img, sar_img):
        opt_img = self.img_resize(opt_img)
        sar_img = self.img_resize(sar_img)

        opt_img = np.float32(opt_img)
        sar_img = np.float32(sar_img)

        cv2.normalize(opt_img, opt_img, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(sar_img, sar_img, 0, 1, cv2.NORM_MINMAX)

        opt_img = (opt_img - self.mean_image) / self.std_image
        sar_img = (sar_img - self.mean_image) / self.std_image
        return opt_img[None, :, :], sar_img[None, :, :]


def create_loaders(config=None):

    img_size = config.imageSize
    test_triplet_aug_op = DataAugment(img_size, config.mean_image, config.std_image)
    train_triplet_aug_op = DataAugment(img_size, config.mean_image, config.std_image)
    typecast_op = transform.TypeCast(mstype.int32)

    train_dataset = ds.GeneratorDataset(
        opt_sar_dataset(root=config.dataroot,
                        train=True),
        column_names=["opt_img", "sar_img", "label"],
        shuffle=False,
        shard_id=config.rank,
        num_shards=config.group_size)

    train_dataset = train_dataset.map(
        operations=train_triplet_aug_op,
        input_columns=['opt_img', 'sar_img'],
        num_parallel_workers=8)
    train_dataset = train_dataset.map(
        operations=typecast_op,
        input_columns=['label'],
        num_parallel_workers=8)

    train_dataset = train_dataset.batch(config.batch_size)

    test_dataset = ds.GeneratorDataset(
        opt_sar_dataset(root=config.dataroot,
                        train=False),
        column_names=["opt_img", "sar_img", "label"],
        shuffle=False)

    test_dataset = test_dataset.map(
        operations=test_triplet_aug_op,
        input_columns=['opt_img', 'sar_img'],
        num_parallel_workers=8)
    test_dataset = test_dataset.map(
        operations=typecast_op,
        input_columns=['label'],
        num_parallel_workers=8)

    test_dataset = test_dataset.batch(config.test_batch_size)

    return train_dataset, test_dataset


def create_evalloaders(config=None):
    img_size = config.imageSize
    test_triplet_aug_op = DataAugment(img_size, config.mean_image, config.std_image)
    typecast_op = transform.TypeCast(mstype.int32)

    eval_dataset = ds.GeneratorDataset(
        opt_sar_dataset(root=config.dataroot,
                        train=False),
        column_names=["opt_img", "sar_img", "label"],
        shuffle=False)

    eval_dataset = eval_dataset.map(
        operations=test_triplet_aug_op,
        input_columns=['opt_img', 'sar_img'],
        num_parallel_workers=8)
    eval_dataset = eval_dataset.map(
        operations=typecast_op,
        input_columns=['label'],
        num_parallel_workers=8)

    eval_dataset = eval_dataset.batch(config.test_batch_size)

    return eval_dataset
