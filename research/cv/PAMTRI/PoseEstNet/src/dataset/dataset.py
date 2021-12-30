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
"""dataset"""
import os
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose

from .veri import VeRiDataset

def create_dataset(cfg, data_dir, is_train=True):
    """create_dataset"""
    device_num, rank_id = _get_rank_info()

    data = VeRiDataset(cfg, data_dir, is_train)

    if is_train:
        if device_num == 1:
            dataset = ds.GeneratorDataset(data, column_names=["input", "target", "target_weight"], \
                num_parallel_workers=1, shuffle=cfg.TRAIN.SHUFFLE, num_shards=1, shard_id=0)
        else:
            dataset = ds.GeneratorDataset(data, column_names=["input", "target", "target_weight"], \
                num_parallel_workers=1, shuffle=cfg.TRAIN.SHUFFLE, num_shards=device_num, shard_id=rank_id)
    else:
        dataset = ds.GeneratorDataset(data, \
            column_names=["input", "target", "target_weight", "center", "scale", "score", "image", \
                "joints", "joints_vis"], num_parallel_workers=1, shuffle=False, num_shards=1, shard_id=0)

    trans = Compose([
        py_vision.ToTensor(),
        py_vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = dataset.map(operations=trans, input_columns="input", num_parallel_workers=8)
    if is_train:
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE, drop_remainder=True, num_parallel_workers=8)
    else:
        dataset = dataset.batch(cfg.TEST.BATCH_SIZE, drop_remainder=True, num_parallel_workers=8)

    if is_train:
        return dataset

    return data, dataset

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE"))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
