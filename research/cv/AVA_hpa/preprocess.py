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
"""preprocess."""

import os
import argparse
import numpy as np

from src.datasets import makeup_dataset


parser = argparse.ArgumentParser(description="preprocess")
parser.add_argument("--data_dir", type=str, default="", help="dataset path")
parser.add_argument("--classes", type=int, default=10, help='class number')
parser.add_argument("--pre_result_dir", type=str, default="./preprocess_Result",
                    help="preprocess data path")
args_opt = parser.parse_args()

if __name__ == '__main__':
    batch_size = 1
    data_dir = args_opt.data_dir

    test_dataset = makeup_dataset(data_dir=data_dir, mode='test', batch_size=1, bag_size=20, classes=args_opt.classes,
                                  num_parallel_workers=8)
    test_dataset.__loop_size__ = 1

    image_path = os.path.join(args_opt.pre_result_dir, "00_data")
    label_path = os.path.join(args_opt.pre_result_dir, "label.npy")
    nslice_path = os.path.join(args_opt.pre_result_dir, "nslice.npy")
    os.makedirs(image_path, exist_ok=True)
    label_list = []
    nslice_list = []
    for i, data in enumerate(test_dataset.create_dict_iterator(output_numpy=True)):
        file_name = "ava_bs" + str(batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(image_path, file_name)
        data["imgs"].tofile(file_path)
        label_list.append(data["labels"])
        nslice_list.append(data["nslice"])
    np.save(os.path.join(label_path), label_list)
    np.save(os.path.join(nslice_path), nslice_list)
    print("=" * 20, "export bin files finished", "=" * 20)
