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

"""post process for 310 inference"""
import os
import argparse
import numpy as np

batch_Size = 1


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="WarpCTC process")
    parser.add_argument("--result_path", default="./outputs", type=str, help="root path of image without noise")
    parser.add_argument('--label_path', default="./label.txt", type=str, help='resized image width')
    args_opt = parser.parse_args()
    return args_opt


def is_eq(pred_lbl, target):
    pred_diff = len(target) - len(pred_lbl)
    if pred_diff > 0:
        pred_lbl.extend([10] * pred_diff)
    return pred_lbl == target


def calcul_acc(y_pred, y):
    correct_num = 0
    total_num = 0
    for b_idx, target in enumerate(y):
        if is_eq(y_pred[b_idx], target):
            correct_num += 1
        total_num += 1
    if total_num == 0:
        raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
    return correct_num / total_num


def get_result(result_path, label_path):
    files = os.listdir(result_path)
    preds = []
    labels = []
    label_dict = {}
    with open(label_path, 'r') as fi:
        lines = fi.readlines()
        for line in lines:
            label_dict[line.split(',')[0].split(".")[0]] = np.array(
                line.replace('\n', '').replace('[', '').replace(']', '').split(',')[1:]).astype(dtype=int).tolist()
    for file in files:
        label = label_dict[file.split(".")[0]]
        labels.append(label)
        resultPath = os.path.join(result_path, file)
        with open(resultPath) as f:
            li = f.readlines()
            for l in li:
                item = l.replace('[', '').replace(']', '').split(',')
                if item == ['']:
                    result = []
                else:
                    result = np.array(item).astype(dtype=int).tolist()
        preds.append(result)
    acc = round(calcul_acc(preds, labels), 3)
    print("Total data: {}, accuracy: {}".format(len(labels), acc))


if __name__ == '__main__':
    args = parse_args()
    get_result(args.result_path, args.label_path)
