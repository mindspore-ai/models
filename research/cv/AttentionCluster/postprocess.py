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
"""postprocess for 310 inference"""
import os
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='postprocess')
    parser.add_argument('--label_path', type=str,
                        default='./preprocess_Result/label/label.bin', help='label directory')
    parser.add_argument('--result_path', type=str,
                        default='./result_Files', help='result_Files directory')

    args_opt = parser.parse_args()

    label_path = args_opt.label_path
    result_path = os.path.join(args_opt.result_path, "input_0.bin")

    label = np.fromfile(label_path, np.float32).reshape(10240, -1)
    label = Tensor(label, mindspore.float32)

    result = np.fromfile(result_path, np.float32).reshape(10240, -1)
    result = Tensor(result, mindspore.float32)

    topk = nn.Top1CategoricalAccuracy()
    topk.clear()
    topk.update(result, label)
    output = topk.eval()
    print("Top1 acc: ", output)
