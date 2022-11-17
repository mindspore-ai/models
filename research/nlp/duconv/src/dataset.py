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
Data loader
"""

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
from mindspore import log as logger

def create_dataset(batch_size, device_num=1, rank=0, num_workers=8, do_shuffle=True,
                   data_file_path=None, use_knowledge=False):
    """create train dataset"""
    if use_knowledge:
        colums_list = ["context_id", "context_segment_id", "context_pos_id", "kn_id", "kn_seq_length", "labels_list"]
    else:
        colums_list = ["context_id", "context_segment_id", "context_pos_id", "labels_list"]

    # apply repeat operations
    data_set = ds.MindDataset(data_file_path,
                              columns_list=colums_list,
                              shuffle=do_shuffle,
                              num_shards=device_num,
                              shard_id=rank,
                              num_parallel_workers=num_workers)

    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="context_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="context_pos_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="context_segment_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="labels_list")
    if use_knowledge:
        data_set = data_set.map(operations=type_cast_op, input_columns="kn_id")
        data_set = data_set.map(operations=type_cast_op, input_columns="kn_seq_length")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set
    