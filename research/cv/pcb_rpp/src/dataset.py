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
Produce the dataset
"""

import multiprocessing
import os
import time

import numpy as np
from mindspore import dataset as ds
from mindspore.common import dtype as mstype
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset.transforms import transforms as C2
from mindspore.dataset.vision import Inter
from mindspore.dataset.vision import transforms as C
from mindspore.mindrecord import FileWriter

from src import datasets
from src.model_utils.config import config


def create_mindrecord_file(data, mindrecord_file, file_num=1):
    """Create MindRecord file."""
    writer = FileWriter(mindrecord_file, file_num)

    schema_json = {
        "image": {"type": "bytes"},
        "fid": {"type": "int32"},
        "pid": {"type": "int32"},
        "camid": {"type": "int32"}
    }
    writer.add_schema(schema_json, "schema_json")

    for fpath, fid, pid, camid in data:
        with open(fpath, 'rb') as f:
            img = f.read()
        row = {"image": img, "fid": fid, "pid": pid, "camid": camid}
        writer.write_raw_data([row])
    writer.commit()


def create_dataset(dataset_name, dataset_path, subset_name, batch_size=32, num_parallel_workers=4, distribute=False):
    """ Create Mindspore dataset

    Args:
        dataset_name: name of one of ('market', 'duke', 'cuhk03') datasets
        dataset_path: path to dataset
        subset_name: name of one of ('train', 'query', 'gallery') subsets
        batch_size: batch size
        num_parallel_workers: the number of readers
        distribute: is distribute training

    Returns:
        Mindspore dataset generator, dataset
    """
    ds.config.set_seed(1)

    subset = datasets.create(dataset_name, root=dataset_path, subset_name=subset_name)
    data = subset.data

    device_num, rank_id = _get_rank_info(distribute)
    mindrecord_dir = os.path.join(config.mindrecord_dir, dataset_name)
    mindrecord_file = os.path.join(mindrecord_dir, subset_name + ".mindrecord")
    if rank_id == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        create_mindrecord_file(data, mindrecord_file)

    while not os.path.exists(mindrecord_file+'.db'):
        time.sleep(5)
    print('Mindrecord found!')
    num_parallel_workers = get_num_parallel_workers(num_parallel_workers)
    is_train = subset_name == "train"
    if device_num == 1:
        data_set = ds.MindDataset(mindrecord_file, columns_list=["image", "fid", "pid", "camid"],
                                  num_parallel_workers=num_parallel_workers, shuffle=is_train)
    else:
        data_set = ds.MindDataset(mindrecord_file, columns_list=["image", "fid", "pid", "camid"], num_shards=device_num,
                                  shard_id=rank_id, num_parallel_workers=num_parallel_workers, shuffle=is_train)

    # map operations on images
    decode_op = C.Decode()
    resize_op = C.Resize([384, 128], Inter.LINEAR)
    flip_op = C.RandomHorizontalFlip(prob=0.5)
    rescale_op = C.Rescale(1.0 / 255.0, 0.0)
    normalize_op = C.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    swap_op = C.HWC2CHW()
    trans = []
    if is_train:
        trans += [decode_op,
                  resize_op,
                  flip_op,
                  rescale_op,
                  normalize_op,
                  swap_op]
    else:
        trans += [decode_op,
                  resize_op,
                  rescale_op,
                  normalize_op,
                  swap_op]

    data_set = data_set.map(operations=trans, input_columns=["image"],
                            num_parallel_workers=num_parallel_workers)

    # map operations on labels
    type_cast_op = C2.TypeCast(mstype.int32)
    squeeze_op = np.squeeze
    trans = [type_cast_op, squeeze_op]
    data_set = data_set.map(operations=trans, input_columns=["fid"],
                            num_parallel_workers=num_parallel_workers)

    data_set = data_set.map(operations=trans, input_columns=["pid"],
                            num_parallel_workers=num_parallel_workers)

    data_set = data_set.map(operations=trans, input_columns=["camid"],
                            num_parallel_workers=num_parallel_workers)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=is_train)
    return data_set, subset


def _get_rank_info(distribute):
    """ Get  device_num and rank_id """
    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    return device_num, rank_id


def get_num_parallel_workers(num_parallel_workers):
    """
    Get num_parallel_workers used in dataset operations.
    If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print(f"The num_parallel_workers {num_parallel_workers} is set too large, now set it {cores}")
            num_parallel_workers = cores
    else:
        print(f"The num_parallel_workers {num_parallel_workers} is invalid, now set it {min(cores, 8)}")
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers
