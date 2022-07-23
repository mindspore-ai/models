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
DAM preprocess script.
"""

import os
import argparse

from mindspore import dataset as ds


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="dam preprocess")
    parser.add_argument('--eval_data_file_path', type=str,
                        default="./data/ubuntu/data_test.mindrecord",
                        help="evaluate data file path")
    parser.add_argument("--eval_batch_size", type=int, default=200,
                        help="Eval batch size, default is 200, 256 for douban")
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')

    args_opt = parser.parse_args()
    return args_opt


if __name__ == "__main__":
    args = parse_args()

    dataset = ds.MindDataset(args.eval_data_file_path,
                             columns_list=["turns", "turn_len", "response", "response_len", "label"],
                             shuffle=False)
    dataset = dataset.batch(batch_size=args.eval_batch_size, drop_remainder=True)
    turns_path = os.path.join(args.result_path, "00_data")
    turn_len_path = os.path.join(args.result_path, "01_data")
    response_path = os.path.join(args.result_path, "02_data")
    response_len_path = os.path.join(args.result_path, "03_data")
    labels_path = os.path.join(args.result_path, "04_data")
    os.makedirs(turns_path)
    os.makedirs(turn_len_path)
    os.makedirs(response_path)
    os.makedirs(response_len_path)
    os.makedirs(labels_path)

    print("=" * 20, "starting export bin files", "=" * 20)
    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "cluener_bs" + str(args.eval_batch_size) + "_" + str(idx) + ".bin"

        turns = data["turns"]
        turn_len = data["turn_len"]
        response = data["response"]
        response_len = data["response_len"]
        labels = data["label"]

        turns_file_path = os.path.join(turns_path, file_name)
        turns.tofile(turns_file_path)

        turn_len_file_path = os.path.join(turn_len_path, file_name)
        turn_len.tofile(turn_len_file_path)

        response_file_path = os.path.join(response_path, file_name)
        response.tofile(response_file_path)

        response_len_file_path = os.path.join(response_len_path, file_name)
        response_len.tofile(response_len_file_path)

        labels_file_path = os.path.join(labels_path, file_name)
        labels.tofile(labels_file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
