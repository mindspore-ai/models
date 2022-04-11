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
"""postprocess_infer data"""

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Postprocess of Hypertext Inference')
parser.add_argument('--result_Path', type=str, help='result path')
parser.add_argument('--label_Path', type=str, help='label file path')
args = parser.parse_args()

cur, total = 0, 0

label = np.loadtxt(args.label_Path, dtype=np.int32).reshape(-1, 1)
predict = np.loadtxt(args.result_Path, dtype=np.int32).reshape(-1, 1)

for i in range(label.shape[0]):
    acc = predict[i] == label[i]
    acc = np.array(acc, dtype=np.float32)
    cur += (np.sum(acc, -1))
    total += len(acc)
print('acc:', cur / total)
