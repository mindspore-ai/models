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
"""Dataset generators."""
import os

from mindspore import dataset as ds
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
import mindspore.dataset.transforms as C2
import mindspore.dataset.vision as C


def create_imagenet(
        dataset_path,
        do_train,
        repeat_num=1,
        input_size=224,
        batch_size=32,
        target="Ascend",
        distribute=False,
        enable_cache=False,
        cache_session_id=None,
):
    """
    Create a train or eval imagenet2012 dataset for cls_hrnet.

    Args:
        dataset_path (str): The path of dataset.
        do_train (bool): Whether dataset is used for train or eval.
        repeat_num(int): The repeat times of dataset. Default: 1
        input_size (int or list): The model input size. Default: 224
        batch_size (int): The batch size of dataset. Default: 32
        target (str): The device target. Default: Ascend
        distribute (bool): Data for distribute or not. Default: False
        enable_cache (bool): Whether tensor caching service is used for eval. Default: False
        cache_session_id (int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        data_set: Inited dataset.
    """
    if target in ["Ascend", "GPU"]:
        device_num, rank_id = _get_rank_info(target)
    else:
        if distribute:
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            rank_id = 0
            device_num = 1

    ds.config.set_prefetch_size(64)
    if device_num == 1:
        data_set = ds.ImageFolderDataset(
            dataset_dir=dataset_path,
            num_parallel_workers=12,
            shuffle=True,
        )
    else:
        data_set = ds.ImageFolderDataset(
            dataset_dir=dataset_path,
            num_parallel_workers=12,
            shuffle=True,
            num_shards=device_num,
            shard_id=rank_id,
        )

    img_scale = 255
    mean = [0.485 * img_scale, 0.456 * img_scale, 0.406 * img_scale]
    std = [0.229 * img_scale, 0.224 * img_scale, 0.225 * img_scale]

    # Define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(input_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(input_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(
        operations=trans,
        input_columns="image",
        num_parallel_workers=12,
        python_multiprocessing=True,
    )
    # Only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")

        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(
            operations=type_cast_op,
            input_columns="label",
            num_parallel_workers=12,
            cache=eval_cache,
        )
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)

    # Apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # Apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info(device_target="Ascend"):
    """
    Get rank size and rank id.
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if device_target in ["Ascend", "GPU"]:
        if rank_size > 1:
            rank_size = get_group_size()
            rank_id = get_rank()
        else:
            rank_size = 1
            rank_id = 0
    else:
        raise ValueError("Unsupported platform.")

    return rank_size, rank_id
