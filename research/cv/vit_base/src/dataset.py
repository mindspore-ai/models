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
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank

from src.config import cifar10_cfg


def create_dataset_cifar10(data_home, device, repeat_num=1, device_num=1, training=True):
    """
    Create a train or eval cifar-10 dataset for vit-base

    Args:
        data_home(str): the path of dataset.
        device(str): device(str): target device platform.
        repeat_num(int): the repeat times of dataset. Default: 1
        device_num(int): num of target devices. Default: 1
        training(bool): whether dataset is used for train or eval.

    Returns:
        dataset
    """

    if device_num > 1:
        rank_size, rank_id = _get_rank_info(device_target=device)
        data_set = ds.Cifar10Dataset(data_home, num_shards=rank_size, shard_id=rank_id, shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(data_home, shuffle=True)

    resize_height = cifar10_cfg.image_height
    resize_width = cifar10_cfg.image_width

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
    if training:
        data_set = data_set.batch(batch_size=cifar10_cfg.batch_size, drop_remainder=True)
    else:
        data_set = data_set.batch(batch_size=1, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info(device_target):
    """get rank size and rank id"""
    if device_target == 'Ascend':
        rank_size = int(os.environ.get("RANK_SIZE", 1))

        if rank_size > 1:
            rank_size = get_group_size()
            rank_id = get_rank()
        else:
            rank_size = rank_id = None
    elif device_target == 'GPU':
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        raise ValueError("Unsupported platform.")

    return rank_size, rank_id
