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
"""create dataset"""
import os

from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision

def create_dataset(data_path, image_size=321, batch_size=32, seed=0, augmentation=True, repeat=True):
    """create dataset"""
    rank_size, rank_id = _get_rank_info()
    data_set = ds.MindDataset(data_path, num_shards=rank_size, shard_id=rank_id)

    columns_to_project = ["data", "label"]
    data_set = data_set.project(columns=columns_to_project)

    # define map
    decode_op = vision.Decode()
    type_cast_op = C.TypeCast(mstype.float32)
    normalize_op = vision.Normalize(mean=[128.0, 128.0, 128.0], std=[128.0, 128.0, 128.0])

    if augmentation:
        crop_resize_op = vision.RandomResizedCrop(image_size, max_attempts=100)
    else:
        crop_resize_op = vision.Resize((image_size, image_size))

    chw_op = vision.HWC2CHW()

    transforms_list = [decode_op, type_cast_op, normalize_op, crop_resize_op, chw_op]

    # map operation
    data_set = data_set.map(operations=transforms_list, input_columns="data")

    ds.config.set_seed(seed)
    if repeat:
        data_set = data_set.repeat()
    data_set = data_set.shuffle(buffer_size=100)

    data_set = data_set.batch(batch_size)

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
