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
"""
Data operations, will be used in train.py and eval.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision


def create_dataset_cifar10(dataset_path, cfg, training, repeat_num=1):
    """Data operations."""
    dataset_path = os.path.join(dataset_path, "cifar-10-batches-bin" if training else "cifar-10-verify-bin")
    if cfg.group_size == 1:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True,
                                     num_shards=cfg.group_size, shard_id=cfg.rank)
    resize_height = cfg.image_height
    resize_width = cfg.image_width

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))  # interpolation default BILINEAR
    rescale_op = vision.Rescale(1.0 / 255.0, 0.0)
    normalize_op = vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply batch operations
    data_set = data_set.batch(batch_size=cfg.batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)
    return data_set


def create_dataset_imagenet(dataset_path, cfg, training, repeat_num=1):
    """
    create a train or eval imagenet2012 dataset for inceptionv2

    Args:
        dataset_path(string): the path of dataset.
        cfg: config
        training(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1

    Returns:
        dataset
    """

    if cfg.group_size == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True,
                                         num_shards=cfg.group_size, shard_id=cfg.rank)
    image_size = cfg.image_height
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if training:
        transform_img = [
            vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
    else:
        transform_img = [
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    transform_label = [C.TypeCast(mstype.int32)]

    data_set = data_set.map(input_columns="image", num_parallel_workers=cfg.work_nums, operations=transform_img,
                            python_multiprocessing=True)
    data_set = data_set.map(input_columns="label", num_parallel_workers=cfg.work_nums, operations=transform_label)

    # apply batch operations
    data_set = data_set.batch(cfg.batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
