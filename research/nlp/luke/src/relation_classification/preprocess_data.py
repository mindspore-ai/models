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
Preprocess data and convert to mindrecord
"""

import json
import os
import pickle

import numpy as np
from mindspore import dataset as ds
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.dataset import transforms as C
from mindspore.mindrecord import FileWriter
from tqdm import tqdm

from src.relation_classification.main import load_examples
from src.utils.utils import create_dir_not_exist


def create_train_dataset(data_file=None, do_shuffle=True, num_shards=1, shard_id=0, batch_size=1,
                         num=None, num_parallel_workers=1):
    """Read train data"""
    dataset = ds.MindDataset(data_file,
                             columns_list=["word_ids", 'word_segment_ids', "word_attention_mask",
                                           "entity_ids", "entity_position_ids",
                                           "entity_segment_ids", "entity_attention_mask",
                                           "label"],
                             shuffle=do_shuffle, num_shards=num_shards, shard_id=shard_id,
                             num_samples=num, num_parallel_workers=num_parallel_workers)
    type_int32 = C.c_transforms.TypeCast(mstype.int32)
    dataset = dataset.map(operations=type_int32, input_columns="word_ids")
    dataset = dataset.map(operations=type_int32, input_columns="word_segment_ids")
    dataset = dataset.map(operations=type_int32, input_columns="word_attention_mask")
    dataset = dataset.map(operations=type_int32, input_columns="entity_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_position_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_segment_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_attention_mask")
    dataset = dataset.map(operations=type_int32, input_columns="label")
    dataset = dataset.batch(batch_size)
    return dataset


def create_eval_dataset(data_file=None, do_shuffle=True, batch_size=1,
                        num=None, num_parallel_workers=1):
    """Read train data"""
    dataset = ds.MindDataset(data_file,
                             columns_list=["word_ids", 'word_segment_ids', "word_attention_mask",
                                           "entity_ids", "entity_position_ids",
                                           "entity_segment_ids", "entity_attention_mask",
                                           "label",
                                           ],
                             shuffle=do_shuffle, num_samples=num, num_parallel_workers=num_parallel_workers)
    type_int32 = C.c_transforms.TypeCast(mstype.int32)
    dataset = dataset.map(operations=type_int32, input_columns="word_ids")
    dataset = dataset.map(operations=type_int32, input_columns="word_segment_ids")
    dataset = dataset.map(operations=type_int32, input_columns="word_attention_mask")
    dataset = dataset.map(operations=type_int32, input_columns="entity_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_position_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_segment_ids")
    dataset = dataset.map(operations=type_int32, input_columns="entity_attention_mask")
    dataset = dataset.map(operations=type_int32, input_columns="label")

    iterator = dataset.create_dict_iterator()
    labels = []
    for item in iterator:
        labels.extend(item["label"].asnumpy().tolist())

    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset, set(labels)


def load_train(args):
    """load train mindrecord"""
    data_dir = os.path.join(args.data + "/mindrecord", "train.mindrecord0")
    rank_size = 1
    rank_id = 0
    if args.distribute:
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
    num_shards = None if rank_size == 1 else rank_size
    shard_id = None if rank_size == 1 else rank_id
    if args.distribute:
        return create_train_dataset(data_dir, batch_size=args.train_batch_size,
                                    num_shards=num_shards,
                                    shard_id=shard_id,
                                    num_parallel_workers=8)
    return create_train_dataset(data_dir, batch_size=args.train_batch_size)


def load_eval(args):
    """load test mindrecord"""
    data_dir = os.path.join(args.data + "/mindrecord", "test.mindrecord0")

    return create_eval_dataset(data_dir, batch_size=args.eval_batch_size)


def change_train_to_mindrecord(args):
    print("change train data to mindrecord")
    dataset = []
    with open(os.path.join(args.data, 'tacred_change', 'train.json'), 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            dataset.append(json.loads(line))
            line = f.readline()
    print("load complete")
    data_dir = os.path.join(args.data, "mindrecord", "train.mindrecord")
    # write mindrecord
    schema_json = {"word_ids": {"type": "int32", "shape": [-1]},
                   "word_segment_ids": {"type": "int32", "shape": [-1]},
                   "word_attention_mask": {"type": "int32", "shape": [-1]},
                   "entity_ids": {"type": "int32", "shape": [-1]},
                   "entity_position_ids": {"type": "int32", "shape": [args.max_entity_length, 30]},
                   "entity_segment_ids": {"type": "int32", "shape": [-1]},
                   "entity_attention_mask": {"type": "int32", "shape": [-1]},
                   "label": {"type": "int32", "shape": [-1]},
                   }

    def get_imdb_data(data):
        """get a iter data"""
        data_list = []
        print('now change，please wait....')
        for each in data:
            data_json = {"word_ids": np.array(each['word_ids'], dtype=np.int32),
                         "word_segment_ids": np.array(each['word_segment_ids'], dtype=np.int32),
                         "word_attention_mask": np.array(each['word_attention_mask'], dtype=np.int32),
                         "entity_ids": np.array(each['entity_ids'], dtype=np.int32),
                         "entity_position_ids": np.array(each['entity_position_ids'], dtype=np.int32),
                         "entity_segment_ids": np.array(each['entity_segment_ids'], dtype=np.int32),
                         "entity_attention_mask": np.array(each['entity_attention_mask'], dtype=np.int32),
                         'label': np.array(each['label'], dtype=np.int32),
                         }

            data_list.append(data_json)
        return data_list

    writer = FileWriter(data_dir, shard_num=4)
    data = get_imdb_data(dataset)
    writer.add_schema(schema_json, "nlp_schema")
    writer.write_raw_data(data)
    writer.commit()
    print("change train data complete")


def change_eval_to_mindrecord(args):
    print("change test data to mindrecord")
    dataset = []
    with open(os.path.join(args.data, 'tacred_change', 'test.json'), 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            dataset.append(json.loads(line))
            line = f.readline()
    print("load complete")
    data_dir = os.path.join(args.data, "mindrecord", "test.mindrecord")
    # write mindrecord
    schema_json = {"word_ids": {"type": "int32", "shape": [-1]},
                   "word_segment_ids": {"type": "int32", "shape": [-1]},
                   "word_attention_mask": {"type": "int32", "shape": [-1]},
                   "entity_ids": {"type": "int32", "shape": [-1]},
                   "entity_position_ids": {"type": "int32", "shape": [args.max_entity_length, 30]},
                   "entity_segment_ids": {"type": "int32", "shape": [-1]},
                   "entity_attention_mask": {"type": "int32", "shape": [-1]},
                   "label": {"type": "int32", "shape": [-1]}
                   }

    def get_imdb_data(data):
        """get a iter data"""
        data_list = []
        print('now change，please wait....')
        for each in data:
            data_json = {"word_ids": np.array(each['word_ids'], dtype=np.int32),
                         "word_segment_ids": np.array(each['word_segment_ids'], dtype=np.int32),
                         "word_attention_mask": np.array(each['word_attention_mask'], dtype=np.int32),
                         "entity_ids": np.array(each['entity_ids'], dtype=np.int32),
                         "entity_position_ids": np.array(each['entity_position_ids'], dtype=np.int32),
                         "entity_segment_ids": np.array(each['entity_segment_ids'], dtype=np.int32),
                         "entity_attention_mask": np.array(each['entity_attention_mask'], dtype=np.int32),
                         "label": np.array(each['label'], dtype=np.int32)
                         }

            data_list.append(data_json)
        return data_list

    writer = FileWriter(data_dir, shard_num=4)
    data = get_imdb_data(dataset)
    writer.add_schema(schema_json, "nlp_schema")
    writer.write_raw_data(data)
    writer.commit()
    print("change test data complete")


class EvalObj:
    """change eval obj"""

    def __init__(self, examples, features, processor):
        """init fun"""
        self.examples = examples
        self.features = features
        self.processor = processor


def save_train(args):
    """save changed train data"""
    print("train")
    train_data = load_examples(args, fold='train')
    with open(os.path.join(args.data, 'tacred_change', 'train.json'), 'w', encoding='utf-8') as f:
        for d in tqdm(train_data):
            f.write(json.dumps(d) + '\n')


def save_eval(args):
    """save changed test data"""
    print('test')

    eval_data, examples, features, processor = load_examples(args, fold='eval')
    # eval_data
    with open(os.path.join(args.data, 'tacred_change', 'test.json'), 'w', encoding='utf-8') as f:
        for d in tqdm(eval_data):
            f.write(json.dumps(d) + '\n')
    eval_obj = EvalObj(examples, features, processor)
    with open(os.path.join(args.data, 'tacred_change', 'test.pickle'), 'wb') as f:
        pickle.dump(eval_obj, f, pickle.HIGHEST_PROTOCOL)


def build_data_change(args):
    create_dir_not_exist(os.path.join(args.data, 'tacred_change'))
    create_dir_not_exist(os.path.join(args.data, 'mindrecord'))
    save_eval(args)
    save_train(args)
    change_eval_to_mindrecord(args)
    change_train_to_mindrecord(args)
