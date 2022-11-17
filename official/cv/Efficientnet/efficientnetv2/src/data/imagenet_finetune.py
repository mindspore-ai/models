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
"""
Data operations, will be used in train.py and eval.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
import numpy as np

from .data_utils.moxing_adapter import sync_data
from .inter import Interpolation


class CutOut:
    """cutout"""

    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, image):
        replace = np.random.uniform(low=0.0, high=1.0, size=image.shape).astype(np.float32)
        image_height = np.shape(image)[0]
        image_width = np.shape(image)[1]
        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = int(np.random.uniform(low=0, high=image_height))
        cutout_center_width = int(np.random.uniform(low=0, high=image_width))
        lower_pad = np.maximum(0, cutout_center_height - self.pad_size)
        upper_pad = np.maximum(0, image_height - cutout_center_height - self.pad_size)
        left_pad = np.maximum(0, cutout_center_width - self.pad_size)
        right_pad = np.maximum(0, image_width - cutout_center_width - self.pad_size)
        cutout_shape = [image_height - (lower_pad + upper_pad),
                        image_width - (left_pad + right_pad)]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = np.pad(np.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
        mask = np.expand_dims(mask, -1)
        mask = np.tile(mask, [1, 1, 3])
        image = np.where(
            np.equal(mask, 0),
            np.ones_like(image, dtype=image.dtype) * replace,
            image)
        return image.astype(np.float32)


class ImageNetFinetune:
    """ImageNet Define"""

    def __init__(self, args, training=True):
        if args.run_modelarts:
            print('Download data.')
            local_data_path = '/cache/data'
            sync_data(args.data_url, local_data_path, threads=128)
            print('Create train and evaluate dataset.')
            train_dir = os.path.join(local_data_path, "train")
            val_ir = os.path.join(local_data_path, "val")
            self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args)
        else:
            train_dir = os.path.join(args.data_url, "train")
            val_ir = os.path.join(args.data_url, "val")
            if training:
                self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args)


def create_dataset_imagenet(dataset_dir, args, repeat_num=1, training=True):
    """
    create a train or eval imagenet2012 dataset for SwinTransformer

    Args:
        dataset_dir(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1

    Returns:
        dataset
    """

    device_num, rank_id = _get_rank_info()
    shuffle = bool(training)
    if device_num == 1 or not training:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers,
                                         shuffle=shuffle)
    else:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers, shuffle=shuffle,
                                         num_shards=device_num, shard_id=rank_id)

    # define map operations
    # BICUBIC: 3
    mean = [127.5, 127.5, 127.5]
    std = [127.5, 127.5, 127.5]
    interpolation = Interpolation[args.interpolation]
    if training:
        cutout_size = args.image_size // 4
        transform_img = [
            vision.Decode(),
            vision.Normalize(mean=mean, std=std),
            vision.Resize(size=(int(args.image_size), int(args.image_size)), interpolation=interpolation),
            vision.RandomHorizontalFlip(prob=0.5),
            CutOut(pad_size=cutout_size),
            vision.HWC2CHW()]
    else:
        # test transform complete
        transform_img = [
            vision.Decode(),
            vision.Normalize(mean=mean, std=std),
            vision.Resize(size=(int(args.image_size / args.crop_pct), int(args.image_size / args.crop_pct)),
                          interpolation=interpolation),
            vision.CenterCrop(args.image_size),
            vision.HWC2CHW()
        ]
    transform_label = C.TypeCast(mstype.int32)
    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
