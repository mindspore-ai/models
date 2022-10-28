# Copyright 2020 Huawei Technologies Co., Ltd
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
""" transform data """
import json

import numpy as np
from mindspore.mindrecord import FileWriter
from sample_process import process_one_example
from tokenization import FullTokenizer
from model_utils.config import config

# Define schema of mindrecord
nlp_schema = {"input_ids": {"type": "int64", "shape": [-1]},
              "attention_mask": {"type": "int64", "shape": [-1]},
              "decoder_input_ids": {"type": "int64", "shape": [-1]},
              "labels": {"type": "int64", "shape": [-1]},
              "decoder_attention_mask": {"type": "int64", "shape": [-1]}
              }


def transform_mind_dataset(file, tokenizer):
    """ transform the dataset """
    mindrecord_file = file.replace('.json', '.mindrecord')
    writer = FileWriter(file_name=mindrecord_file, shard_num=1)
    writer.add_schema(nlp_schema)
    f = open(file, 'r', encoding='utf-8')
    origin_data = json.load(f)
    num = 0
    data = []
    for i in origin_data:
        num += 1
        document = origin_data[i]['document']
        summary = origin_data[i]['summary']
        input_ids, attention_mask = process_one_example(tokenizer, document, max_seq_len=512)
        decoder_input_ids, decoder_attention_mask = process_one_example(tokenizer, summary, max_seq_len=64)
        labels = decoder_input_ids.copy()
        # padding id from i to -100
        for j, l in enumerate(labels):
            if l == 1:
                labels[j] = -100
        sample = {
            'input_ids': np.array(input_ids, dtype=np.int64),
            'attention_mask': np.array(attention_mask, dtype=np.int64),
            'decoder_input_ids': np.array(decoder_input_ids, dtype=np.int64),
            'labels': np.array(labels, dtype=np.int64),
            'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.int64)
        }
        data.append(sample)
        if num % 100 == 0:
            writer.write_raw_data(data)
            data = []
    if data:
        writer.write_raw_data(data)
    writer.commit()


if __name__ == '__main__':
    vocab_file = config.vocab_file_path
    _tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    train_data_json_dir = os.path.join(config.data_path, 'train.json')
    validation_data_json_dir = os.path.join(config.data_path, 'validation.json')
    test_data_json_dir = os.path.join(config.data_path, 'test.json')
    transform_mind_dataset(train_data_json_dir, tokenizer=_tokenizer)
    transform_mind_dataset(validation_data_json_dir, tokenizer=_tokenizer)
    transform_mind_dataset(test_data_json_dir, tokenizer=_tokenizer)
