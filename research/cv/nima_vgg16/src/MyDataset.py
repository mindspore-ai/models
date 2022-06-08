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
"""AVA dataset builders."""

import os

import cv2
import numpy as np
from mindspore import dataset as ds
from mindspore import dtype as mstype
from mindspore.dataset.transforms import transforms as t_ct
from mindspore.dataset.vision import Inter
from mindspore.dataset.vision import transforms as v_ct


class Dataset:
    """
    Simple image dataset

    Args:
        image_list: list of image paths for dataset.
        label_list: list of labels for images.
    """
    def __init__(self, image_list, label_list):
        super(Dataset, self).__init__()
        self.imgs = image_list
        self.labels = label_list

    def __getitem__(self, index):
        """Get sample"""
        img = cv2.imread(self.imgs[index])
        return img, self.labels[index]

    def __len__(self):
        """Len of dataset."""
        return len(self.imgs)


def score_lst(lst):
    """evaluate lst score."""
    lst = np.array(lst).astype(int)
    res = lst / sum(lst)
    return res


def create_dataset(args, data_mode='train'):
    """Create MindSpore dataset."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rank_id = args.rank
    rank_size = args.device_num
    if data_mode == 'train':
        with open(args.train_label_path, 'r') as f:
            datafile = f.readlines()
        transform_img = t_ct.Compose([
            v_ct.Resize([args.bf_crop_size, args.bf_crop_size], Inter.BICUBIC),
            v_ct.RandomCrop(args.image_size),
            v_ct.RandomHorizontalFlip(prob=0.5),
            v_ct.Normalize(mean=mean, std=std),
            v_ct.HWC2CHW()])
    else:
        with open(args.val_label_path, 'r') as f:
            datafile = f.readlines()
        transform_img = t_ct.Compose([
            v_ct.Resize([args.image_size, args.image_size], Inter.BICUBIC),
            v_ct.RandomHorizontalFlip(prob=0.5),
            v_ct.Normalize(mean=mean, std=std),
            v_ct.HWC2CHW()])
    transform_label = t_ct.TypeCast(mstype.float32)

    save_image_list = [os.path.join(args.data_path, i.split(',')[1]) for i in datafile]
    save_label_list = [score_lst(i.split(',')[2:12]) for i in datafile]
    dataset = Dataset(save_image_list, save_label_list)
    if data_mode == 'train':
        if rank_size == 1:
            de_dataset = ds.GeneratorDataset(dataset, column_names=["image", "label"],
                                             shuffle=True, num_parallel_workers=args.num_parallel_workers)
        else:
            de_dataset = ds.GeneratorDataset(dataset, column_names=["image", "label"],
                                             shuffle=True, num_parallel_workers=args.num_parallel_workers,
                                             num_shards=rank_size, shard_id=rank_id)
        drop_remainder = True
    else:
        de_dataset = ds.GeneratorDataset(dataset, column_names=["image", "label"],
                                         shuffle=False, num_parallel_workers=args.num_parallel_workers)
        drop_remainder = False
    de_dataset = de_dataset.map(input_columns="image", operations=transform_img)
    de_dataset = de_dataset.map(input_columns="label", operations=transform_label)

    de_dataset = de_dataset.batch(args.batch_size,
                                  drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(1)
    steps_per_epoch = de_dataset.get_dataset_size()
    return de_dataset, steps_per_epoch
