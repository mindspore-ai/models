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
import mindspore.dataset.engine as de
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size


def create_dataset(dataset_path, do_train, repeat_num=1, infer_910=False, device_id=0,\
    batch_size=32, distribute: bool = False, target: str = "GPU"):
    """
    create a train or evaluate cifar10 dataset for WideResnet
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    num_parallel_workers = 5

    if target == "Ascend" and infer_910:
        rank_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
            rank_id = 0

    if do_train:
        dataset_path = os.path.join(dataset_path, 'train')
    else:
        dataset_path = os.path.join(dataset_path, 'eval')

    if device_num == 1:
        ds = de.Cifar10Dataset(dataset_path,
                               num_parallel_workers=num_parallel_workers,
                               shuffle=True)
    else:
        if do_train:
            ds = de.Cifar10Dataset(dataset_path,
                                   num_parallel_workers=num_parallel_workers,
                                   shuffle=True,
                                   num_shards=device_num,
                                   shard_id=rank_id)
        else:
            ds = de.Cifar10Dataset(dataset_path)

    trans = []
    if do_train:
        trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(operations=type_cast_op,
                input_columns="label",
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=trans,
                input_columns="image",
                num_parallel_workers=num_parallel_workers)

    ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.repeat(repeat_num)

    return ds
