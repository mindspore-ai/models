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
"""dataset"""
import os
import copy
import json
from pathlib import Path
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms.transforms import Compose
from mindspore.communication.management import get_rank

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
        vision.ToTensor(),
        vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), is_hwc=False)
    ])

    dataset = dataset.map(operations=trans, input_columns="input", num_parallel_workers=8)
    if is_train:
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE, drop_remainder=True, num_parallel_workers=8)
    else:
        dataset = dataset.batch(cfg.TEST.BATCH_SIZE, drop_remainder=True, num_parallel_workers=8)

    if is_train:
        return dataset

    return data, dataset

def get_label(cfg, data_dir):
    """
    get label
    """
    lable_path = os.path.join(data_dir, 'annot/image_test.json')

    if not os.path.isfile(lable_path):
        os.mknod(lable_path)
        data = VeRiDataset(cfg, data_dir, False).db

        label = {}
        for i in range(data.__len__()):
            out = copy.deepcopy(data[i])
            label['{}'.format(i)] = out['image']

        label_json_path = Path(lable_path)
        with label_json_path.open('w') as dst_file:
            json.dump(label, dst_file)

    return lable_path

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE"))
    print(rank_size)
    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))

        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
