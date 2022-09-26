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

'''
Ernie preprocess script.
'''

import os
import argparse
from src.tokenization import CscTokenizer

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="soft-masked bert preprocess")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Eval batch size, default is 2")
    parser.add_argument("--eval_data_file_path", type=str, default="./dataset/dev.json",
                        help="Data path, it is better to use absolute path")
    parser.add_argument('--result_path', type=str, default='./preprocess_result/', help='result path')
    parser.add_argument('--device_num', type=int, default=1, help='device num')
    parser.add_argument('--rank_id', type=int, default=0, help='rank id')
    parser.add_argument('--vocab_path', type=str, default='./src/bert-base-chinese-vocab.txt', help='vocab path')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    tokenizer = CscTokenizer(fp=args.eval_data_file_path, device_num=args.device_num, rank_id=args.rank_id, \
                             max_seq_len=512, vocab_path=args.vocab_path)
    ds = tokenizer.get_token_ids(args.eval_batch_size)
    print(ds.dataset_size)
    wrong_ids_path = os.path.join(args.result_path, "00_data")
    original_tokens_path = os.path.join(args.result_path, "01_data")
    original_tokens_mask_path = os.path.join(args.result_path, "02_data")
    correct_tokens_path = os.path.join(args.result_path, "03_data")
    correct_tokens_mask_path = os.path.join(args.result_path, "04_data")
    original_token_type_ids_path = os.path.join(args.result_path, "05_data")
    correct_token_type_ids_path = os.path.join(args.result_path, "06_data")
    os.makedirs(wrong_ids_path)
    os.makedirs(original_tokens_path)
    os.makedirs(original_tokens_mask_path)
    os.makedirs(correct_tokens_path)
    os.makedirs(correct_tokens_mask_path)
    os.makedirs(original_token_type_ids_path)
    os.makedirs(correct_token_type_ids_path)

    for idx, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        wrong_ids = data["wrong_ids"]
        original_tokens = data["original_tokens"]
        original_tokens_mask = data["original_tokens_mask"]
        correct_tokens = data["correct_tokens"]
        correct_tokens_mask = data["correct_tokens_mask"]
        original_token_type_ids = data["original_token_type_ids"]
        correct_token_type_ids = data["correct_token_type_ids"]

        file_name = "batch_" + str(args.eval_batch_size) + "_" + str(idx) + ".bin"
        wrong_ids_file_path = os.path.join(wrong_ids_path, file_name)
        wrong_ids.tofile(wrong_ids_file_path)
        original_tokens_file_path = os.path.join(original_tokens_path, file_name)
        original_tokens.tofile(original_tokens_file_path)
        original_tokens_mask_file_path = os.path.join(original_tokens_mask_path, file_name)
        original_tokens_mask.tofile(original_tokens_mask_file_path)
        correct_tokens_file_path = os.path.join(correct_tokens_path, file_name)
        correct_tokens.tofile(correct_tokens_file_path)
        correct_tokens_mask_file_path = os.path.join(correct_tokens_mask_path, file_name)
        correct_tokens_mask.tofile(correct_tokens_mask_file_path)
        original_token_type_ids_file_path = os.path.join(original_token_type_ids_path, file_name)
        original_token_type_ids.tofile(original_token_type_ids_file_path)
        correct_token_type_ids_file_path = os.path.join(correct_token_type_ids_path, file_name)
        correct_token_type_ids.tofile(correct_token_type_ids_file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
