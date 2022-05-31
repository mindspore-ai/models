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
Produce the dataset
"""

import os
from multiprocessing import cpu_count
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank, get_group_size


def create_dataset_cifar10(cfg, data_path, batch_size=32, status="train", target="Ascend",
                           num_parallel_workers=8):
    """
    create dataset for train or test
    """

    ds.config.set_prefetch_size(64)
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()

    if target != "Ascend" or device_num == 1:
        cifar_ds = ds.Cifar10Dataset(data_path, shuffle=True)
    else:
        cifar_ds = ds.Cifar10Dataset(data_path, num_parallel_workers=num_parallel_workers,
                                     shuffle=True, num_shards=device_num, shard_id=rank_id)
    rescale = 1.0 / 255.0
    shift = 0.0
    # cfg = alexnet_cifar10_cfg

    resize_op = CV.Resize((cfg.image_height, cfg.image_width))
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if status == "train":
        random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
        random_horizontal_op = CV.RandomHorizontalFlip()
    channel_swap_op = CV.HWC2CHW()
    typecast_op = C.TypeCast(mstype.int32)
    cifar_ds = cifar_ds.map(input_columns="label", operations=typecast_op,
                            num_parallel_workers=1)
    if status == "train":
        compose_op = [random_crop_op, random_horizontal_op, resize_op, rescale_op, normalize_op, channel_swap_op]
    else:
        compose_op = [resize_op, rescale_op, normalize_op, channel_swap_op]
    cifar_ds = cifar_ds.map(input_columns="image", operations=compose_op, num_parallel_workers=num_parallel_workers)

    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    return cifar_ds


def create_dataset_imagenet(cfg, dataset_path, batch_size=32, repeat_num=1, training=True,
                            num_parallel_workers=16, shuffle=None, sampler=None, class_indexing=None):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """

    device_num, rank_id = _get_rank_info()
    # cfg = alexnet_imagenet_cfg

    if device_num == 1:
        num_parallel_workers = 96
        if num_parallel_workers > cpu_count():
            num_parallel_workers = cpu_count()
    else:
        ds.config.set_numa_enable(True)
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=4,
                                     shuffle=shuffle, sampler=sampler, class_indexing=class_indexing,
                                     num_shards=device_num, shard_id=rank_id)

    assert cfg.image_height == cfg.image_width, "imagenet_cfg.image_height not equal imagenet_cfg.image_width"
    image_size = cfg.image_height

    # define map operations
    transform_img = []
    if training:
        transform_img = [
            CV.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            CV.RandomHorizontalFlip(prob=0.5)
        ]
    else:
        transform_img = [
            CV.Decode(),
            CV.Resize((256, 256)),
            CV.CenterCrop(image_size)
        ]

    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers,
                            operations=transform_img)

    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    if repeat_num > 1:
        data_set = data_set.repeat(repeat_num)

    return data_set


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
