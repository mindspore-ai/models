# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Data operations, will be used in train.py."""
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset as ds


def create_dataset(batch_size=1, data_file_path=None, rank_size=1, rank_id=0,
                   repeat_count=1, do_shuffle=True, drop_remainder=True, num_samples=None):
    """create dataset """
    type_cast_op = C.TypeCast(mstype.int32)
    dataset = ds.MindDataset([data_file_path],
                             columns_list=['input_ids', 'attention_mask', 'decoder_input_ids',
                                           'decoder_attention_mask', 'labels'],
                             shuffle=do_shuffle, num_shards=rank_size, shard_id=rank_id,
                             num_samples=num_samples)  # ,num_samples=128
    count = 0
    for _ in dataset.create_dict_iterator():
        count += 1
    print("Got {} samples".format(count))
    dataset = dataset.map(operations=type_cast_op, input_columns='input_ids')
    dataset = dataset.map(operations=type_cast_op, input_columns='attention_mask')
    dataset = dataset.map(operations=type_cast_op, input_columns='decoder_input_ids')
    dataset = dataset.map(operations=type_cast_op, input_columns='decoder_attention_mask')
    dataset = dataset.map(operations=type_cast_op, input_columns='labels')
    dataset = dataset.repeat(repeat_count)
    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


def create_test_dataset(batch_size=1, data_file_path=None, rank_size=1, rank_id=0,
                        repeat_count=1, do_shuffle=True, drop_remainder=True, num_samples=None):
    """ create test dataset """
    type_cast_op = C.TypeCast(mstype.int32)
    dataset = ds.MindDataset([data_file_path],
                             columns_list=['input_ids', 'attention_mask', 'decoder_input_ids',
                                           'decoder_attention_mask', 'labels', 'summary'],
                             shuffle=do_shuffle, num_shards=rank_size, shard_id=rank_id,
                             num_samples=num_samples)  # ,num_samples=128
    count = 0
    for _ in dataset.create_dict_iterator():
        count += 1
    print("Got {} samples".format(count))
    dataset = dataset.map(operations=type_cast_op, input_columns='input_ids')
    dataset = dataset.map(operations=type_cast_op, input_columns='attention_mask')
    dataset = dataset.map(operations=type_cast_op, input_columns='decoder_input_ids')
    dataset = dataset.map(operations=type_cast_op, input_columns='decoder_attention_mask')
    dataset = dataset.map(operations=type_cast_op, input_columns='labels')
    dataset = dataset.map(input_columns='summary')
    dataset = dataset.repeat(repeat_count)
    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset
