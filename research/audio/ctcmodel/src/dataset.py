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

"""Dataset preprocessing."""

import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.common.dtype as mstype


def create_dataset(path, isTrain, batch_size, num_shards=1, shard_id=0):
    '''create dataset'''
    dataset = ds.MindDataset(path, columns_list=["feature", "masks", "label", "seq_len"], num_parallel_workers=4,
                             shuffle=isTrain, num_shards=num_shards, shard_id=shard_id)
    type_cast_op = C.TypeCast(mstype.int32)
    type_cast_op2 = C.TypeCast(mstype.float32)
    dataset = dataset.map(operations=type_cast_op2, input_columns="feature")
    dataset = dataset.map(operations=type_cast_op2, input_columns="masks")
    dataset = dataset.map(operations=type_cast_op, input_columns="label")
    dataset = dataset.map(operations=type_cast_op, input_columns="seq_len")
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset
