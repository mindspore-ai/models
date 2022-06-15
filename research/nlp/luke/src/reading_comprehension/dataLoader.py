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
"""data loader file"""
import os
import pickle5 as pickle

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C

from src.model_utils.moxing_adapter import get_device_num
from src.model_utils.moxing_adapter import get_rank_id


def create_dataset(data_file=None, do_shuffle=True, device_num=1, rank=0, batch_size=1,
                   num=None, num_parallel_workers=1):
    """Read train data"""
    dataset = ds.MindDataset(data_file,
                             columns_list=["word_ids", 'word_segment_ids', "word_attention_mask",
                                           "entity_ids", "entity_position_ids",
                                           "entity_segment_ids", "entity_attention_mask",
                                           "start_positions", "end_positions"],
                             shuffle=do_shuffle, num_shards=device_num, shard_id=rank,
                             num_samples=num, num_parallel_workers=num_parallel_workers)
    type_int32 = C.TypeCast(mstype.int32)
    dataset = dataset.map(operations=type_int32, input_columns="word_ids")
    dataset = dataset.map(operations=type_int32, input_columns="word_segment_ids")
    dataset = dataset.map(operations=type_int32, input_columns="word_attention_mask")
    dataset = dataset.map(operations=type_int32, input_columns="entity_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_position_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_segment_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_attention_mask")
    dataset = dataset.map(operations=type_int32, input_columns="start_positions")
    dataset = dataset.map(operations=type_int32, input_columns="end_positions")

    dataset = dataset.batch(batch_size)
    return dataset


def create_eval_dataset(data_file=None, do_shuffle=True, device_num=1, rank=0, batch_size=1,
                        num=None, num_parallel_workers=1):
    """Read train data"""
    dataset = ds.MindDataset(data_file,
                             columns_list=["word_ids", 'word_segment_ids', "word_attention_mask",
                                           "entity_ids", "entity_position_ids",
                                           "entity_segment_ids", "entity_attention_mask",
                                           "example_indices"],
                             shuffle=do_shuffle, num_shards=device_num, shard_id=rank,
                             num_samples=num, num_parallel_workers=num_parallel_workers)
    type_int32 = C.TypeCast(mstype.int32)
    dataset = dataset.map(operations=type_int32, input_columns="word_ids")
    dataset = dataset.map(operations=type_int32, input_columns="word_segment_ids")
    dataset = dataset.map(operations=type_int32, input_columns="word_attention_mask")
    dataset = dataset.map(operations=type_int32, input_columns="entity_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_position_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_segment_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_attention_mask")
    dataset = dataset.map(operations=type_int32, input_columns="example_indices")

    dataset = dataset.batch(batch_size)
    return dataset


def load_train(args):
    """load train mindrecord"""
    data_dir = os.path.join(args.data + "/mindrecord", "train.mindrecord0")
    if args.modelArts or args.duoka:
        return create_dataset(data_dir, batch_size=args.train_batch_size, device_num=get_device_num(),
                              rank=get_rank_id(),
                              num_parallel_workers=8)
    return create_dataset(data_dir, batch_size=args.train_batch_size)


def load_eval(args):
    """load eval data"""
    # eval_data
    data_dir = os.path.join(args.data + "/mindrecord", "eval_data.mindrecord0")
    data_set = create_eval_dataset(data_dir, batch_size=args.eval_batch_size, do_shuffle=False)
    # other
    examples = None
    features = None
    processor = None

    with open(args.data + '/squad_change/eval_obj.pickle', 'rb') as f:
        eval_obj = pickle.load(f)
    examples = eval_obj.examples
    features = eval_obj.features
    processor = eval_obj.processor

    return data_set, examples, features, processor
