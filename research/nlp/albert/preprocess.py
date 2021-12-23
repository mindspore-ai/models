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

'''
albert preprocess script.
'''

import os
import argparse
import pickle
from src.dataset import create_classification_dataset, create_squad_dataset


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="ernie preprocess")
    parser.add_argument("--task_type", type=str, default="false",
                        choices=["mnli", "sst2", "squadv1"],
                        help="Eval task type, default is msra_ner")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')
    parser.add_argument('--eval_json_path', type=str, default="", help='eval json path')
    parser.add_argument('--vocab_file_path', type=str, default="", help='vocab file path')
    parser.add_argument('--spm_model_file', type=str, default="", help='spm model file path')

    args_opt = parser.parse_args()

    if args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    args.eval_batch_size = 1
    if args.task_type == 'mnli' or args.task_type == 'sst2':
        ds = create_classification_dataset(batch_size=args.eval_batch_size,
                                           repeat_count=1,
                                           data_file_path=args.eval_data_file_path,
                                           do_shuffle=(args.eval_data_shuffle.lower() == "true"))

        label_path = os.path.join(args.result_path, "03_data")
        os.makedirs(label_path)
    elif args.task_type == 'squadv1':
        from src import tokenization
        from src.squad_utils import read_squad_examples, convert_examples_to_features

        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file_path,
                                               do_lower_case=True,
                                               spm_model_file=args.spm_model_file)

        eval_examples = read_squad_examples(args.eval_json_path, False)
        if not os.path.exists(args.eval_data_file_path):
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=False,
                output_fn=None,
                do_lower_case=True)
            with open(args.eval_data_file_path, "wb") as fout:
                pickle.dump(eval_features, fout)
        else:
            with open(args.eval_data_file_path, "rb") as fin:
                eval_features = pickle.load(fin)
        ds = create_squad_dataset(batch_size=args.eval_batch_size, repeat_count=1,
                                  data_file_path=eval_features, is_training=False,
                                  do_shuffle=(args.eval_data_shuffle.lower() == "true"))

        unique_path = os.path.join(args.result_path, "03_data")
        os.makedirs(unique_path)
    else:
        raise ValueError("dataset not supported, support: [mnli, sst2, squadv1]")

    ids_path = os.path.join(args.result_path, "00_data")
    mask_path = os.path.join(args.result_path, "01_data")
    token_path = os.path.join(args.result_path, "02_data")
    os.makedirs(ids_path)
    os.makedirs(mask_path)
    os.makedirs(token_path)

    for idx, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        input_ids = data["input_ids"]
        input_mask = data["input_mask"]
        segment_ids = data["segment_ids"]

        if args.task_type == 'mnli' or args.task_type == 'sst2':
            label_ids = data["label_ids"]
        else:
            unique_ids = data["unique_ids"]

        file_name = args.task_type + "_batch_" + str(args.eval_batch_size) + "_" + str(idx) + ".bin"
        ids_file_path = os.path.join(ids_path, file_name)
        input_ids.tofile(ids_file_path)

        mask_file_path = os.path.join(mask_path, file_name)
        input_mask.tofile(mask_file_path)

        token_file_path = os.path.join(token_path, file_name)
        segment_ids.tofile(token_file_path)

        if args.task_type == 'mnli' or args.task_type == 'sst2':
            label_file_path = os.path.join(label_path, file_name)
            label_ids.tofile(label_file_path)
        elif args.task_type == 'squadv1':
            unique_file_path = os.path.join(unique_path, file_name)
            unique_ids.tofile(unique_file_path)
        else:
            raise ValueError("dataset not supported, support: [mnli, sst2, squadv1]")

    print("=" * 20, "export bin files finished", "=" * 20)
