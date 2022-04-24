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

'''squad data precess'''
import argparse
import os
import pickle
from create_squad_data import read_squad_examples, convert_examples_to_features
import numpy as np
import tokenization


def parse_args():
    """set parameters."""
    parser = argparse.ArgumentParser(description="bert preprocess")
    parser.add_argument("--vocab_path", type=str,
                        default="../data/config/vocab.txt")
    parser.add_argument("--spm_model_file", type=str,
                        default="../data/input", help="the path of convert dataset.")
    parser.add_argument("--dev_path", type=str, default="../data/dev.json")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="sentence length, default is 128.")
    parser.add_argument("--output_path", type=str,
                        default="../data/input", help="the path of convert dataset.")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")

    args = parser.parse_args()
    return args


def get_all_path(output_path):
    """
    Args:
        output_path: save path of convert dataset
    Returns:
        the path of ids, mask, token, label
    """
    ids_path = os.path.join(output_path, "00_data")  # input_ids
    mask_path = os.path.join(output_path, "01_data")  # input_mask
    token_path = os.path.join(output_path, "02_data")  # segment_ids
    label_path = os.path.join(output_path, "03_data")  # unique_id

    for path in [ids_path, mask_path, token_path, label_path]:
        os.makedirs(path, 0o755, exist_ok=True)

    return ids_path, mask_path, token_path, label_path


def run():
    '''main function'''
    args = parse_args()
    input_ids, input_mask, segment_ids, unique_id = get_all_path(
        args.output_path)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_path, do_lower_case=True, spm_model_file=args.spm_model_file)
    eval_examples = read_squad_examples(args.dev_path, False)
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

    for i in range(len(eval_features)):
        file_name = "squadv1" + "_batch_1_" + str(i) + ".bin"
        ids_file_path = os.path.join(input_ids, file_name)
        np.array(eval_features[i].input_ids,
                 dtype=np.int32).tofile(ids_file_path)

        input_mask_path = os.path.join(input_mask, file_name)
        np.array(eval_features[i].input_mask,
                 dtype=np.int32).tofile(input_mask_path)

        segment_ids_path = os.path.join(segment_ids, file_name)
        np.array(eval_features[i].segment_ids,
                 dtype=np.int32).tofile(segment_ids_path)

        unique_id_path = os.path.join(unique_id, file_name)
        np.array(eval_features[i].unique_id,
                 dtype=np.int32).tofile(unique_id_path)


if __name__ == "__main__":
    run()
