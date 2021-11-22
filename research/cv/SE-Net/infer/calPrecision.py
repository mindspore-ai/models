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

parser = argparse.ArgumentParser(description='SE_net calcul acc')
parser.add_argument("--result_path", type=str, required=True, default='', help="result file path")
parser.add_argument("--data_path", type=str, required=True, default='', help="data path")
args = parser.parse_args()


def get_top5_acc(top_arg, gt_class):
    sub_count = 0
    if gt_class in top_arg:
        sub_count += 1
    return sub_count


def get_label(data_path):
    img_label = {}
    dirs = os.listdir(data_path)
    dirs = sorted(dirs)
    for class_num, dir_ in enumerate(dirs):
        files = os.listdir(os.path.join(data_path, dir_))
        for file_ in files:
            img_label[file_.split('.')[0]] = class_num
    return img_label


def cal_acc_imagenet(result_path, data_path):
    """ calcul acc """
    img_label = get_label(data_path)
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    files = os.listdir(result_path)
    for file_ in files:
        full_file_path = os.path.join(result_path, file_)
        if os.path.isfile(full_file_path):
            result = np.loadtxt(full_file_path, dtype=np.float32)
            gt_classes = int(img_label[file_.split('_1')[0]])
            top1_output = result[0]
            top5_output = result
            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            top5_correct += get_top5_acc(top5_output, gt_classes)
            img_tot += 1
    acc1 = 100 * top1_correct / img_tot
    acc5 = 100 * top5_correct / img_tot
    print('total={}, top1_correct={}, acc={:.2f}%'.format(img_tot, top1_correct, acc1))
    print('total={}, top5_correct={}, acc={:.2f}%'.format(img_tot, top5_correct, acc5))


if __name__ == '__main__':
    cal_acc_imagenet(args.result_path, args.data_path)
