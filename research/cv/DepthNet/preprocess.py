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
preprocess script
"""
import os
import argparse
from src.data_loader import create_test_dataset

parser = argparse.ArgumentParser(description="postprocess for googlenet")
parser.add_argument("--output_path", type=str, required=True, help="result file path")
parser.add_argument("--dataset_path", type=str, required=True, help="test data path")
args = parser.parse_args()
test_batch_size = 1


def preprocess(test_data_path, result_path):
    test_dataset = create_test_dataset(test_data_path, batch_size=test_batch_size)
    img_path = os.path.join(result_path, "img_data")
    label_path = os.path.join(result_path, "label")
    os.makedirs(img_path)
    os.makedirs(label_path)
    idx = 0
    for data in test_dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_data = data["rgb"]
        img_label = data["ground_truth"]

        file_name = "DepthNet_nyu_" + str(test_batch_size) + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)
        idx += 1

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == "__main__":
    preprocess(args.dataset_path, args.output_path)
