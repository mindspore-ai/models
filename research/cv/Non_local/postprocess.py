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

"""postprocess for 310 inference"""
import os
import argparse
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='postprocess')
    parser.add_argument('--label_path', type=str,
                        default='./preprocess_Result/label/label.bin', help='label directory')
    parser.add_argument('--result_path', type=str,
                        default='./result_Files', help='result_Files directory')
    parser.add_argument('--dataset', type=str,
                        default='kinetics', help='dataset')
    args_opt = parser.parse_args()

    label_path = args_opt.label_path

    label = np.fromfile(label_path, np.int64).reshape(-1, 1)
    label = Tensor(label)
    if args_opt.dataset == 'kinetics':
        label = np.fromfile(label_path, np.int64).reshape(-1, 1)
        label = Tensor(label)
        result_files = os.listdir(args_opt.result_path)
        count = 0
        results = []
        for file_name in result_files:
            count = max(int(file_name.split('_')[0]), count)
        for i in range(count + 1):
            result = np.fromfile(os.path.join(args_opt.result_path, str(i) + '_0.bin'), np.float32).reshape(-1, 400)
            results.append(result)
        results = np.stack(results).reshape(-1, 400)
        results = Tensor(results)
    else:
        label = np.fromfile(label_path, np.int32).reshape(-1, 1)
        label = Tensor(label)
        result_path = os.path.join(args_opt.result_path, "input_0.bin")
        results = np.fromfile(result_path, np.float32).reshape(-1, 10)
    topk = nn.Top1CategoricalAccuracy()
    topk.clear()
    topk.update(results, label)
    output = topk.eval()
    print("Top1 acc: ", output)
