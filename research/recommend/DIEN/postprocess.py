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

"""postprocess"""

import os
import time
import numpy as np

from src.config import parse_args

args_opt = parse_args()


def calc_auc(raw_arr):
    """clac_auc"""
    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if abs(record[1] - 1.) < 0.000001:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if abs(record[1] - 1.) < 0.000001:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


def get_acc(result_path, target_path):
    """get_acc"""
    steps = 0
    stored_arr = []
    acc_sum = 0
    time_start = time.time()
    for i in range(len(os.listdir(target_path))):
        result_file = os.path.join(result_path, "DIEN_data_bs" + args_opt.dataset_type + str(i) + "_0.bin")
        target_file = os.path.join(target_path, "DIEN_data_bs" + args_opt.dataset_type + str(i) + ".bin")
        pred_y = np.fromfile(result_file, dtype=np.float32).reshape(128, 2)
        target = np.fromfile(target_file, dtype=np.float32).reshape(128, 2)
        y_hat_1 = pred_y[:, 0].tolist()
        target_2 = target[:, 0].tolist()
        for y, t in zip(y_hat_1, target_2):
            stored_arr.append([y, t])
        y_hat = np.round(pred_y)
        acc = (y_hat == target).sum() / 256
        acc_sum += acc
        steps += 1
    time_end = time.time()
    test_acc = acc_sum / steps
    test_auc = calc_auc(stored_arr)
    print('acc:{0}  test_auc:{1}'.format(test_acc, test_auc))
    print('spend_time:', time_end - time_start)


def main():
    result_path = args_opt.binary_files_path
    target_path = args_opt.target_binary_files_path

    get_acc(result_path, target_path)


if __name__ == "__main__":
    main()
