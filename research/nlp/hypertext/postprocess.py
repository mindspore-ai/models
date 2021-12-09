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
"""preprocess data"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Postprocess of Hypertext Inference')
parser.add_argument('--result_Path', type=str, default='./result_Files',
                    help='result path')
parser.add_argument('--label_Path', default='./result_Files', type=str,
                    help='label file path')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
args = parser.parse_args()

dirs = os.listdir(args.label_Path)
cur, total = 0, 0
print('---------- start cal acc ----------')
for file in dirs:
    label = np.fromfile(os.path.join(args.label_Path, file), dtype=np.int32)
    file_name = file.split('.')[0]
    idx = file_name.split('_')[-1]
    predict_file_name = "hypertext_ids_bs" + str(args.batch_size) + "_" + str(idx) + "_0.bin"
    predict_file = os.path.join(args.result_Path, predict_file_name)
    predict = np.fromfile(predict_file, dtype=np.int32)
    acc = predict == label
    acc = np.array(acc, dtype=np.float32)
    cur += (np.sum(acc, -1))
    total += len(acc)
print('acc:', cur / total)
