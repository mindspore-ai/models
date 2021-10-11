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
#################lstm postprocess########################
"""
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="atae-lstm postprocess")
parser.add_argument("--label_dir", type=str, required=True, help="label directory")
parser.add_argument("--result_dir", type=str, required=True, help="result directory")
args = parser.parse_args()

if __name__ == '__main__':
    file_names = []
    for root, dirs, files in os.walk(args.result_dir):
        file_names = files

    file_num = len(file_names)
    correct = 0
    for f in file_names:
        label_path = os.path.join(args.label_dir, f)
        result_path = os.path.join(args.result_dir, f)

        label_numpy = np.fromfile(label_path, np.float32).reshape([1, 3])
        polarity_label = np.argmax(label_numpy)
        result_numpy = np.fromfile(result_path, np.float32).reshape([1, 3])
        polarity_result = np.argmax(result_numpy)
        if polarity_result == polarity_label:
            correct += 1

    acc = correct / float(file_num)
    print("\n---accuracy:", acc, "---\n")
