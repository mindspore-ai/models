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
'''post process for 310 inference'''
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='postprocess for googlenet')
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_file", type=str, required=True, help="label file")
args = parser.parse_args()


def cal_acc_imagenet(result_path, label_file):
    """cal_acc_imagenet"""
    img_tot = 0
    top1_correct = 0

    files = os.listdir(result_path)
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).argmax()
            idx_num = file.split("_")[0].split("s")[1]
            gt_classes = np.fromfile(os.path.join(label_file, 'label{}.bin'.format(idx_num)), dtype=np.int32)

            if result == gt_classes:
                top1_correct = top1_correct + 1
            img_tot += 1

    acc1 = 100.0 * top1_correct / img_tot
    print('after allreduce eval: top1_correct={}, tot={}, acc={:.2f}%'.format(top1_correct, img_tot, acc1))

if __name__ == "__main__":
    cal_acc_imagenet(args.result_path, args.label_file)
