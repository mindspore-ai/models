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
"""
mindspore utils, will be used in train.py and eval.py
"""
import os
import random

import mindspore.dataset as ds
from mindspore import context, save_checkpoint
from mindspore.common import set_seed
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.context import ParallelMode

from src.common.logger import Logger


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def _get_rank_info():
    """
    get rank info
    :return: rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


def my_save_checkpoint(save_obj, ckpt_file_name):
    """
    save once
    :param save_obj: save_obj
    :param ckpt_file_name: ckpt_file_name
    """
    _, rank_id = _get_rank_info()
    if rank_id == 0:
        save_checkpoint(save_obj, ckpt_file_name)


class MSUtils:
    """
    utils for initialize and prepare dataloader
    """
    @staticmethod
    def initialize(device="CPU", device_id=0):
        """
        :param device: support GPU/CPU/Ascend
        """
        set_seed(1)
        ds.config.set_seed(1)
        random.seed(1)
        if device == 'CPU':
            context.set_context(mode=context.PYNATIVE_MODE, device_target=device)
        else:
            print("initialize, rank %d / %d, device_id: %d" % (get_rank_id() + 1, get_device_num(), device_id))

            device_num = get_device_num()
            context.set_context(mode=context.GRAPH_MODE, device_target=device, save_graphs=False)
            if device_num > 1:
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
                context.set_context(device_id=device_id)
                init()
            else:
                context.set_context(device_id=device_id)

    @staticmethod
    def prepare_dataloader(dataset, column_names, batch_size=None, num_workers=1, is_shuffle=False):
        """
        prepare dataloader
        :param dataset: dataset
        :param column_names: column_names
        :param batch_size: batch_size
        :param num_workers: worker numbers
        :param is_shuffle: shuffle or not
        :return: dict iterator
        """
        device_num, rank_id = _get_rank_info()

        dataloader = ds.GeneratorDataset(dataset, column_names, shuffle=is_shuffle,
                                         num_shards=device_num, shard_id=rank_id, num_parallel_workers=num_workers)
        if batch_size is not None:
            dataloader = dataloader.batch(batch_size, drop_remainder=False, num_parallel_workers=num_workers)

        if dataloader.get_dataset_size() == 0:
            Logger().critical('The dataset is empty!')
        return dataloader.create_dict_iterator()
