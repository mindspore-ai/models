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

import argparse
import os
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

parser = argparse.ArgumentParser(description='postprocess for tcn')
parser.add_argument("--dataset_name", type=str, default="permuted_mnist", help="result file path")
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_path", type=str, required=True, help="label file")
args = parser.parse_args()


def cal_acc_premuted_mnist(result_path, label_path):
    """post process of premuted mnist for 310 inference"""
    img_total = 0
    totalcorrect = 0
    files = os.listdir(result_path)
    for file in files:
        batch_size = int(file.split('tcn_premuted_mnist')[1].split('_')[0])
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape((batch_size, 10))
            label_file = os.path.join(label_path, file)
            gt_classes = np.fromfile(label_file, dtype=np.int32)
            top1_output = np.argmax(result, (-1))
            correct = np.equal(top1_output, gt_classes).sum()
            totalcorrect += correct
            img_total += batch_size

    acc1 = 100.0 * totalcorrect / img_total
    print('acc={:.4f}%'.format(acc1))


def cal_acc_adding_problem(result_path, label_path):
    """post process of adding problem for 310 inference"""
    files = os.listdir(result_path)
    label_name = os.listdir(label_path)[0]
    label_file = os.path.join(label_path, label_name)
    error = nn.MSE()
    mse_loss = []
    for file in files:
        full_file_path = os.path.join(result_path, file)
        result = Tensor(np.fromfile(full_file_path, dtype=np.float32).reshape((1000, 1)))
        label = Tensor(np.fromfile(label_file, dtype=np.float32))
        error.clear()
        error.update(result, label)
        result = error.eval()
        mse_loss.append(result)
    acc = sum(mse_loss) / len(mse_loss)
    print('myloss={}'.format(acc))


if __name__ == "__main__":
    if args.dataset_name == "permuted_mnist":
        cal_acc_premuted_mnist(args.result_path, args.label_path)
    elif args.dataset_name == "adding_problem":
        cal_acc_adding_problem(args.result_path, args.label_path)
