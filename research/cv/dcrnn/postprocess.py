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

import os
import argparse
import numpy as np

def cal_acc():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--label-path', type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path
    label_path = args.label_path
    result_path = os.path.abspath(os.path.dirname(
        __file__)) + "/scripts/result_Files"
    file_num = len(os.listdir(result_path))
    acc_sum = 0
    for i in range(0, file_num):
        datapath = os.path.join(data_path, 'data' + '_' + str(i) + '_0' + '.bin')
        data = np.fromfile(datapath, np.float32)
        data = data.reshape(12, 64, 207)
        labelpath = os.path.join(label_path, 'label' + '_' + str(i) + '.bin')
        label = np.fromfile(labelpath, np.float32)
        label = label.reshape(12, 64, 207)
        mask = label/(label.mean())
        loss = np.abs(label - data)
        loss = loss * mask
        loss[0] = 0
        print("single step:", i, "mae is", abs(loss.mean())/12)
        acc_sum += abs(loss.mean())
    print("file number", file_num)
    accuracy_top1 = acc_sum / file_num
    accuracy_top1 = accuracy_top1 / 12
    print('eval result: ', accuracy_top1)

if __name__ == '__main__':
    cal_acc()
