# Copyright 2021 Huawei Technologies Co., Ltd
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
process the cluener json
"""
import os
import argparse
import numpy as np

from src.data.data_set import DataSet
from infer.util.register import import_modules


sstcfg = {
    "dataset_reader": {
        "train_reader": {
            "name": "train_reader",
            "type": "OneSentClassifyReaderEn",
            "fields": [
                {
                    "name": "qid",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "label",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "text_a",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "roberta_en.vocab.bpe",
                            "bpe_json_file": "roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json"
                    }
                }
            ],
            "config": {
                "data_path": "data/en/finetune/SST-2/train",
                "shuffle": True,
                "batch_size": 1,
                "epoch": 10,
                "sampling_rate": 1.0
            }
        },
        "test_reader": {
            "name": "test_reader",
            "type": "OneSentClassifyReaderEn",
            "fields": [
                {
                    "name": "qid",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "text_a",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "roberta_en.vocab.bpe",
                            "bpe_json_file": "roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json",
                        "other": ""
                    }
                },
                {
                    "name": "label",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                }
            ],
            "config": {
                "data_path": "data/en/finetune/SST-2/test",
                "shuffle": False,
                "batch_size": 1,
                "epoch": 1,
                "sampling_rate": 1.0
            }
        },
        "dev_reader": {
            "name": "dev_reader",
            "type": "OneSentClassifyReaderEn",
            "fields": [
                {
                    "name": "qid",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "label",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "text_a",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "roberta_en.vocab.bpe",
                            "bpe_json_file": "roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json",
                        "other": ""
                    }
                }
            ],
            "config": {
                "data_path": "data/en/finetune/SST-2/dev",
                "shuffle": False,
                "batch_size": 1,
                "epoch": 1,
                "sampling_rate": 1.0
            }
        }
    },
    "model": {
        "type": "RobertaOneSentClassificationEn",
        "embedding": {
            "type": "ErnieTokenEmbedding",
            "emb_dim": 1024,
            "use_fp16": False,
            "config_path": "./model_files/config/roberta_large_en.config.json",
            "other": ""
        },
        "optimization": {
            "learning_rate": 3e-5,
        }
    }

}


def dataset_reader_from_params(params_dict):
    """
    :param params_dict:
    :return:
    """
    reader = DataSet(params_dict)
    reader.build()
    return reader

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="preprocess")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_data_file_path", type=str, default="data/",
                        help="Data path, it is better to use absolute path")
    parser.add_argument('--result_path', type=str, default='data/', help='result path')

    args_opt = parser.parse_args()

    if args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    return args_opt


def run():
    """
    convert infer json to bin, each sentence is one file bin
    """
    args = parse_args()
    _params = sstcfg
    import_modules()
    dataset_reader_params_dict = _params.get("dataset_reader")
    dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)
    train_wrapper = dataset_reader.dev_reader.data_generator()
    ids_path = os.path.join(args.result_path, "00_data")
    mask_path = os.path.join(args.result_path, "02_data")
    token_path = os.path.join(args.result_path, "01_data")
    label_path = os.path.join(args.result_path, "03_data")
    os.makedirs(ids_path)
    os.makedirs(mask_path)
    os.makedirs(token_path)
    os.makedirs(label_path)
    idx = 0
    for i in train_wrapper():
        input_ids = np.array(i[2], dtype=np.int32)
        input_mask = np.array(i[5], dtype=np.int32)
        token_type_id = np.array(i[3], dtype=np.int32)
        label_ids = np.array(i[1], dtype=np.int32)

        file_name = "senta_batch_1_" + str(idx) + ".bin"
        ids_file_path = os.path.join(ids_path, file_name)
        input_ids.tofile(ids_file_path)

        mask_file_path = os.path.join(mask_path, file_name)
        input_mask.tofile(mask_file_path)

        token_file_path = os.path.join(token_path, file_name)
        token_type_id.tofile(token_file_path)

        label_file_path = os.path.join(label_path, file_name)
        label_ids.tofile(label_file_path)
        idx += 1
    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == "__main__":
    run()
