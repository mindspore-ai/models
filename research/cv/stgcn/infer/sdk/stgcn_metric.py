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
"""compute accuracy"""
import os
import sys
import numpy as np

def run():
    """compute acc"""
    if len(sys.argv) == 3:
        # the path to store the results path
        result_file = sys.argv[1]
        # the path to store the label path
        label_file = sys.argv[2]
    else:
        print("Please enter target file result folder | ground truth label file | result json file folder | "
              "result json file name, such as ./result val_label.txt . result.json")
        exit(1)
    if not os.path.exists(result_file):
        print("Target file folder does not exist.")

    if not os.path.exists(label_file):
        print("Label file does not exist.")

    predcitions = np.loadtxt(result_file)
    labels = np.loadtxt(label_file)
    mae, mape, mse = [], [], []
    for predcition, label in zip(predcitions, labels):
        d = np.abs(predcition - label)
        mae += d.tolist()
        mape += (d / label).tolist()
        mse += (d ** 2).tolist()

    MAE = np.array(mae).mean()
    MAPE = np.array(mape).mean()
    RMSE = np.sqrt(np.array(mse).mean())
    print(f'MAE {MAE:.2f} | MAPE {MAPE*100:.2f} | RMSE {RMSE:.2f}')

if __name__ == '__main__':
    run()
