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
"""post process for 310 inference"""
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='PostProcess args')
parser.add_argument('--result_path', type=str, required=True, help='Dataset path')
parser.add_argument('--ori_path', type=str, required=True, help='Train output path')

args_opt = parser.parse_args()


if __name__ == '__main__':
    result_path = args_opt.result_path
    ori_path = args_opt.ori_path
    count = 0

    result_file = os.listdir(result_path)
    ori_path_file = os.listdir(ori_path)

    assert len(result_file) == len(ori_path_file)

    total_num = len(result_file)

    for i in range(total_num):
        ori_label_name = os.path.join(ori_path, 'sop_' + str(i) + '.bin')
        result_label = os.path.join(result_path, 'sop_' + str(i) + '_0.bin')
        ori_label = np.fromfile(ori_label_name, np.int64)
        result_label = np.argmax(np.fromfile(result_label, np.float32), axis=0)
        print("Start processing", ori_label_name)
        if ori_label == result_label:
            count += 1

    acc = 100 * count / total_num
    print("=" * 20, "Convert bin files finished", "=" * 20)
    print("Accuracy is", round(acc, 2), "%")
