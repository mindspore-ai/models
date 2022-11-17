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
data processor file.
"""

import os
import argparse
import numpy as np
import dataset
from dataset import Tuple, Pad, Stack
from tokenizer import FullTokenizer

def get_all_path(output_path):
    """
    Args:
        output_path: save path of convert dataset
    Returns:
        the path of ids, mask, token, label
    """
    ids_path = os.path.join(output_path, "00_data")
    mask_path = os.path.join(output_path, "01_data")
    token_path = os.path.join(output_path, "02_data")
    label_path = os.path.join(output_path, "03_data")
    for path in [ids_path, mask_path, token_path, label_path]:
        os.makedirs(path, 0o755, exist_ok=True)

    return ids_path, mask_path, token_path, label_path

TASK_CLASSES = {
    'atis_intent': dataset.ATIS_DID,
    'mrda': dataset.MRDA,
    'swda': dataset.SwDA
}

def data_save_to_file(data_file_path=None, vocab_file_path='bert-base-uncased-vocab.txt', \
        output_path=None, task_name=None, mode="test", max_seq_length=128):
    """data save to mindrecord file."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_ids, output_mask, output_token, output_label = get_all_path(output_path)
    dataset_class = TASK_CLASSES[task_name]
    tokenizer = FullTokenizer(vocab_file=vocab_file_path, do_lower_case=True)
    task_dataset = dataset_class(data_file_path, mode=mode)
    applid_data = []
    print(task_name + " " + mode + " data process begin")
    dataset_len = len(task_dataset)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=0),  # input
        Pad(axis=0, pad_val=0),  # mask
        Pad(axis=0, pad_val=0),  # segment
        Stack(dtype='int64')  # label
    ): fn(samples)
    for idx, example in enumerate(task_dataset):
        if idx % 1000 == 0:
            print("Reading example %d of %d" % (idx, dataset_len))
        data_example = dataset_class.convert_example(example=example, \
                tokenizer=tokenizer, max_seq_length=max_seq_length)
        applid_data.append(data_example)

    applid_data = batchify_fn(applid_data)
    input_ids, input_mask, segment_ids, label_ids = applid_data

    for idx in range(dataset_len):
        if idx % 1000 == 0:
            print("Processing example %d of %d" % (idx, dataset_len))
        file_name = task_name + "_" + str(idx) + ".bin"
        ids_file_path = os.path.join(output_ids, file_name)
        np.array(input_ids[idx], dtype=np.int32).tofile(ids_file_path)
        mask_file_path = os.path.join(output_mask, file_name)
        np.array(input_mask[idx], dtype=np.int32).tofile(mask_file_path)

        token_file_path = os.path.join(output_token, file_name)
        np.array(segment_ids[idx], dtype=np.int32).tofile(token_file_path)

        label_file_path = os.path.join(output_label, file_name)
        np.array(label_ids[idx], dtype=np.int32).tofile(label_file_path)

    print(task_name + " " + mode + " data process end, " + "total:" + str(dataset_len))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dgu dataset process")
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.")
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="The directory where the dataset will be load.")
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        help="The directory where the vocab will be load.")
    parser.add_argument(
        "--mode",
        default="test",
        type=str,
        help="The mode will be do.[test, infer]")
    parser.add_argument(
        "--max_seq_len",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization for trainng. ")
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="The directory where the mindrecord dataset file will be save.")

    args = parser.parse_args()
    data_save_to_file(data_file_path=args.data_path, vocab_file_path=args.vocab_file, output_path=args.output_path, \
            task_name=args.task_name, mode=args.mode, max_seq_length=args.max_seq_len)
