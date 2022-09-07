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
Create train or eval dataset.
"""
import os
from enum import Enum

import numpy as np
import mindspore.dataset as ds

class DataType(Enum):
    """
    Enumerate supported dataset format.
    """
    MINDRECORD = 1

def _get_mindrecord_dataset(directory, train_mode=True, epochs=1, batch_size=1000,
                            line_per_sample=1000, rank_size=None, rank_id=None):
    """
    Get dataset with mindrecord format.

    Args:
        directory (str): Dataset directory.
        train_mode (bool): Whether dataset is use for train or eval (default=True).
        epochs (int): Dataset epoch size (default=1).
        batch_size (int): Dataset batch size (default=1000).
        line_per_sample (int): The number of sample per line (default=1000).
        rank_size (int): The number of device, not necessary for single device (default=None).
        rank_id (int): Id of device, not necessary for single device (default=None).

    Returns:
        Dataset.
    """
    # not file name, just prefix to identify train or test data
    file_prefix_name = 'train_input_part.mindrecord' if train_mode else 'test_input_part.mindrecord'
    file_suffix_name = '000' if train_mode else '00'
    shuffle = train_mode

    if rank_size is not None and rank_id is not None:
        data_set = ds.MindDataset(os.path.join(directory, file_prefix_name + file_suffix_name),
                                  columns_list=['feat_ids', 'feat_vals', 'label'],
                                  num_shards=rank_size, shard_id=rank_id, shuffle=shuffle,
                                  num_parallel_workers=8)
    else:
        data_set = ds.MindDataset(os.path.join(directory, file_prefix_name + file_suffix_name),
                                  columns_list=['feat_ids', 'feat_vals', 'label'],
                                  shuffle=shuffle, num_parallel_workers=8)
    data_set = data_set.batch(int(batch_size / line_per_sample), drop_remainder=True)
    data_set = data_set.map(operations=(lambda x, y, z: (np.array(x).flatten().reshape(batch_size, 26),
                                                         np.log(np.array(y).flatten().reshape(batch_size, 13) + 1), # deal with numerical features
                                                         np.array(z).flatten().reshape(batch_size, 1))),
                            input_columns=['feat_ids', 'feat_vals', 'label'],
                            num_parallel_workers=8)
    data_set = data_set.repeat(epochs)
    return data_set


def create_dataset(directory, train_mode=True, epochs=1, batch_size=1000,
                   data_type=DataType.MINDRECORD, line_per_sample=1000,
                   rank_size=None, rank_id=None):
    """
    Get dataset.

    Args:
        directory (str): Dataset directory.
        train_mode (bool): Whether dataset is use for train or eval (default=True).
        epochs (int): Dataset epoch size (default=1).
        batch_size (int): Dataset batch size (default=1000).
        data_type (DataType): The type of dataset which is one of H5, TFRECORE, MINDRECORD (default=MINDRECORD).
        line_per_sample (int): The number of sample per line (default=1000).
        rank_size (int): The number of device, not necessary for single device (default=None).
        rank_id (int): Id of device, not necessary for single device (default=None).

    Returns:
        Dataset.
    """
    return _get_mindrecord_dataset(directory, train_mode, epochs,
                                   batch_size, line_per_sample,
                                   rank_size, rank_id)
