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
create train or eval dataset.
"""
import os
import mindspore
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as py_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms2
from mindspore.dataset.vision import Inter
from mindspore.communication.management import init, get_rank, get_group_size
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_num, get_rank_id

def create_dataset1(dataset_path, do_train, repeat_num=1, batch_size=32,
                    target="Ascend", distribute=False):
    """
    create a train or evaluate cifar10 dataset for resnet50
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
    ds.config.set_prefetch_size(64)
    if do_train:
        usage = "train"
        transform = py_transforms2.Compose([py_transforms.ToPIL(),
                                            py_transforms.RandomHorizontalFlip(),
                                            py_transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
                                            py_transforms.ToTensor()])
    else:
        usage = "test"
        transform = py_transforms2.Compose([py_transforms.ToPIL(),
                                            py_transforms.ToTensor()])
    if device_num == 1:
        dataset = ds.Cifar10Dataset(dataset_path, usage=usage, num_parallel_workers=8, shuffle=True)
    else:
        dataset = ds.Cifar10Dataset(dataset_path, usage=usage, num_parallel_workers=8,
                                    shuffle=True, num_shards=device_num, shard_id=rank_id)
    type_cast_op = C2.TypeCast(mindspore.int32)
    dataset = dataset.map(operations=transform, input_columns="image")
    dataset = dataset.map(operations=type_cast_op, input_columns="label")
    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    dataset = dataset.repeat(repeat_num)
    return dataset

def create_dataset2(dataset_path, do_train, repeat_num=1, batch_size=32, train_image_size=224, eval_image_size=224,
                    target="Ascend", distribute=False, enable_cache=False, cache_session_id=None):
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
    ds.config.set_prefetch_size(64)
    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), interpolation=Inter.BICUBIC),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(eval_image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=12)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)
    return data_set

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if config.device_target == "Ascend":
        if rank_size > 1:
            rank_size = get_device_num()
            rank_id = get_rank_id()
        else:
            rank_size = 1
            rank_id = 0
    else:
        if rank_size > 1:
            rank_size = get_group_size()
            rank_id = get_rank()
        else:
            rank_size = 1
            rank_id = 0

    return rank_size, rank_id
