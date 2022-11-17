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
Ernie preprocess script.
'''

import os
import argparse
import numpy as np
from mindspore.dataset import MindDataset


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="duconv preprocess")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')

    args_opt = parser.parse_args()

    if args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    ds = MindDataset(
        args.eval_data_file_path,
        num_parallel_workers=8,
        shuffle=True)

    context_id_path = os.path.join(args.result_path, "00_data")
    context_segment_id_path = os.path.join(args.result_path, "01_data")
    context_pos_id_path = os.path.join(args.result_path, "02_data")
    kn_id_path = os.path.join(args.result_path, "03_data")
    kn_seq_length_path = os.path.join(args.result_path, "04_data")
    label_path = os.path.join(args.result_path, "05_data")


    os.makedirs(context_id_path)
    os.makedirs(context_segment_id_path)
    os.makedirs(context_pos_id_path)
    os.makedirs(kn_id_path)
    os.makedirs(kn_seq_length_path)
    os.makedirs(label_path)

    for idx, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        context_ids = data["context_id"].astype(np.int32)
        context_segment_ids = data["context_segment_id"].astype(np.int32)
        context_pos_ids = data["context_pos_id"].astype(np.int32)
        kn_ids = data["kn_id"].astype(np.int32)
        kn_seq_lengths = data["kn_seq_length"].astype(np.int32)
        label_ids = data["labels_list"].astype(np.int32)

        file_name = "duconv_batch_" + str(args.eval_batch_size) + "_" + str(idx) + ".bin"
        context_file_path = os.path.join(context_id_path, file_name)
        context_ids.tofile(context_file_path)

        context_segment_file_path = os.path.join(context_segment_id_path, file_name)
        context_segment_ids.tofile(context_segment_file_path)

        context_pos_file_path = os.path.join(context_pos_id_path, file_name)
        context_pos_ids.tofile(context_pos_file_path)

        kn_file_path = os.path.join(kn_id_path, file_name)
        kn_ids.tofile(kn_file_path)

        kn_seq_length_file_path = os.path.join(kn_seq_length_path, file_name)
        kn_seq_lengths.tofile(kn_seq_length_file_path)

        label_file_path = os.path.join(label_path, file_name)
        label_ids.tofile(label_file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
