# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
create train or eval dataset.
"""
import os
from multiprocessing import cpu_count
import mindspore as ms
import mindspore.dataset as ds
import mindspore.communication as comm

THREAD_NUM = 12 if cpu_count() >= 12 else 8


def create_dataset1(dataset_path, do_train, device_num=1, batch_size=32, target="Ascend"):
    """
    create a train or evaluate cifar10 dataset for mobilenet
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """
    if device_num == 1:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=THREAD_NUM, shuffle=True)
    else:
        device_num, rank_id = _get_rank_info()
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=THREAD_NUM, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            ds.vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            ds.vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        ds.vision.Resize((224, 224)),
        ds.vision.Rescale(1.0 / 255.0, 0.0),
        ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ds.vision.HWC2CHW()
    ]

    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=THREAD_NUM)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=THREAD_NUM)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


def create_dataset2(dataset_path, do_train, device_num=1, batch_size=32, target="Ascend"):
    """
    create a train or eval imagenet2012 dataset for mobilenet

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=THREAD_NUM, shuffle=True)
    else:
        device_num, rank_id = _get_rank_info()
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=THREAD_NUM, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)

    image_size = 224
    # Computed from random subset of ImageNet training images
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            ds.vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            ds.vision.RandomHorizontalFlip(prob=0.5),
            ds.vision.Normalize(mean=mean, std=std),
            ds.vision.HWC2CHW()
        ]
    else:
        trans = [
            ds.vision.Decode(),
            ds.vision.Resize(256),
            ds.vision.CenterCrop(image_size),
            ds.vision.Normalize(mean=mean, std=std),
            ds.vision.HWC2CHW()
        ]

    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=THREAD_NUM)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=THREAD_NUM)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = comm.get_group_size()
        rank_id = comm.get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
