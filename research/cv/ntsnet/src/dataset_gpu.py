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

"""ntsnet dataset"""
import os

import mindspore.dataset as de
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter

from src.config_gpu import config

mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def create_dataset_train(train_path, batch_size):
    """create train dataset"""
    device_num, rank_id = _get_rank_info()

    if device_num == 1:
        train_data_set = de.ImageFolderDataset(train_path, num_parallel_workers=8, shuffle=True)
    else:
        train_data_set = de.ImageFolderDataset(train_path, num_parallel_workers=8,
                                               shuffle=True, num_shards=device_num, shard_id=rank_id)

    # define map operations
    transform_img = [
        vision.Decode(),
        vision.Resize(config.crop_pct_size, Inter.BILINEAR),
        vision.RandomCrop(config.input_size),
        vision.RandomHorizontalFlip(),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]
    train_data_set = train_data_set.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
    train_data_set = train_data_set.batch(batch_size, drop_remainder=True)
    return train_data_set


def create_dataset_test(test_path, batch_size):
    """create test dataset"""
    test_data_set = de.ImageFolderDataset(test_path, shuffle=False)
    # define map operations
    transform_img = [
        vision.Decode(),
        vision.Resize(config.crop_pct_size, Inter.BILINEAR),
        vision.CenterCrop(config.input_size),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]
    test_data_set = test_data_set.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
    test_data_set = test_data_set.batch(batch_size, drop_remainder=True)
    return test_data_set


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
