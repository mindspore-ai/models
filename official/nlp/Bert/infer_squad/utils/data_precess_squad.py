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
    parser.add_argument("--vocab_path", type=str, default="../data/config/vocab.txt")
    parser.add_argument("--dev_path", type=str, default="../data/dev.json")
    parser.add_argument("--max_seq_len", type=int, default=128, help="sentence length, default is 128.")
    parser.add_argument("--output_path", type=str, default="../data/input", help="the path of convert dataset.")

    args = parser.parse_args()
    return args


def get_all_path(output_path):
    """
    Args:
        output_path: save path of convert dataset
    Returns:
        the path of ids, mask, token, label
    """
    ids_path = os.path.join(output_path, "00_data")#input_ids
    mask_path = os.path.join(output_path, "01_data")#input_mask
    token_path = os.path.join(output_path, "02_data")#segment_ids
    label_path = os.path.join(output_path, "03_data")#unique_id
    tokens_path = os.path.join(output_path, "04_data")#04_tokens
    token_to_orig_map_path = os.path.join(output_path, "05_data")#token_to_orig_map
    token_is_max_context_path = os.path.join(output_path, "06_data")#token_is_max_context
    doc_tokens_path = os.path.join(output_path, "07_data")#doc_tokens
    qas_id_path = os.path.join(output_path, "08_data")#qas_id
    example_index_path = os.path.join(output_path, "09_data")#example_index

    for path in [ids_path, mask_path, token_path, label_path, tokens_path, token_to_orig_map_path,
                 token_is_max_context_path, doc_tokens_path, qas_id_path, example_index_path]:
        os.makedirs(path, 0o755, exist_ok=True)

    return ids_path, mask_path, token_path, label_path, tokens_path, token_to_orig_map_path, \
           token_is_max_context_path, doc_tokens_path, qas_id_path, example_index_path


def run():
    '''main function'''
    args = parse_args()
    input_ids, input_mask, segment_ids, unique_id, tokens, token_to_orig_map, token_is_max_context, doc_tokens, \
    qas_id, example_index = get_all_path(args.output_path)
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_path, do_lower_case=True)
    eval_examples = read_squad_examples(args.dev_path, False)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        output_fn=None,
        vocab_file=args.vocab_path)

    for i in range(len(eval_examples)):
        file_name = "squad_bs" + "_" + str(i) + ".bin"
        qas_id_path = os.path.join(qas_id, file_name)
        with open(qas_id_path, "wb") as f:
            pickle.dump(eval_examples[i].qas_id, f)

        doc_tokens_path = os.path.join(doc_tokens, file_name)
        with open(doc_tokens_path, "wb") as f:
            pickle.dump(eval_examples[i].doc_tokens, f)

    for i in range(len(eval_features)):
        file_name = "squad_bs" + "_" + str(i) + ".bin"
        ids_file_path = os.path.join(input_ids, file_name)
        np.array(eval_features[i].input_ids, dtype=np.int32).tofile(ids_file_path)

        input_mask_path = os.path.join(input_mask, file_name)
        np.array(eval_features[i].input_mask, dtype=np.int32).tofile(input_mask_path)

        segment_ids_path = os.path.join(segment_ids, file_name)
        np.array(eval_features[i].segment_ids, dtype=np.int32).tofile(segment_ids_path)

        unique_id_path = os.path.join(unique_id, file_name)
        np.array(eval_features[i].unique_id, dtype=np.int32).tofile(unique_id_path)

        tokens_path = os.path.join(tokens, file_name)
        with open(tokens_path, "wb") as f:
            pickle.dump(eval_features[i].tokens, f)

        token_to_orig_map_path = os.path.join(token_to_orig_map, file_name)
        with open(token_to_orig_map_path, "wb") as f:
            pickle.dump(eval_features[i].token_to_orig_map, f)

        token_is_max_context_path = os.path.join(token_is_max_context, file_name)
        with open(token_is_max_context_path, "wb") as f:
            pickle.dump(eval_features[i].token_is_max_context, f)

        example_index_path = os.path.join(example_index, file_name)
        np.array(eval_features[i].example_index, dtype=np.int32).tofile(example_index_path)


if __name__ == "__main__":
    run()
