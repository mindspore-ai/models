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
Data operations, will be used in train.py and eval.py
"""
import mindspore.dataset as ds

def open_mindrecord_dataset(dataset_path, do_train, rank, group_size,
                            columns_list,
                            num_parallel_workers=8, batch_size=125,
                            drop_remainder=False, shuffle=True):
    """
    open a train or val dataset in mindrecord format

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        rank(int): The shard ID within num_shards.
        group_size(int): Number of shards that the dataset should be divided into.
        columns_list(list): the columns for the dataset.
        num_parallel_workers(int): the number of parallel workers (Default:8).
        batch_size(int): the batch size for dataset (Default:125).
        drop_remainder(bool): whether to drop the remainder in dataset (Default:False).
        shuffle(bool): whether to shuffle the dataset (Default:True).

    Returns:
        dataset
    """
    if group_size == 1  or not do_train:
        data_set = ds.MindDataset(dataset_path, columns_list=columns_list,
                                  num_parallel_workers=num_parallel_workers,
                                  shuffle=shuffle)
        print(dataset_path)
    else:
        data_set = ds.MindDataset(dataset_path, columns_list=columns_list,
                                  num_parallel_workers=num_parallel_workers,
                                  shuffle=shuffle,
                                  num_shards=group_size, shard_id=rank)
        print(dataset_path, ' group_size = ', group_size, ' rank = ', rank)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=drop_remainder)
    return data_set
