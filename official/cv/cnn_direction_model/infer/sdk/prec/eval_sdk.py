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
""" eval_sdk.py """
import os
import numpy as np


def read_file_list(input_file):
    """
    :param infer file content:
        0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
        1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
        ...
    :return image path list
    """
    image_file_l = []
    if not os.path.exists(input_file):
        print('input file does not exists.')
    with open(input_file, "r") as fs:
        for line in fs.readlines():
            if len(line) > 10:
                line = line.strip('\n').split('\t')[0].replace('\\', '/')
                image_file_l.append(line)
    return image_file_l


def cal_acc(result, gt_classes):
    img_total = len(gt_classes)
    top1_correct = 0

    top1_output = np.argmax(result, (-1))

    t1_correct = np.equal(top1_output, gt_classes).sum()
    top1_correct += t1_correct

    acc1 = 100.0 * top1_correct / img_total
    print('top1_correct={}, total={}, acc={:.2f}%'.format(top1_correct, img_total, acc1))


if __name__ == "__main__":
    results_path = '../result/infer_results.txt'
    infer_file = '../../data/image/annotation_test.txt'
    results = np.loadtxt(results_path)

    file_list = read_file_list(infer_file)
    labels = [int(file.split('_')[-1].split('.')[0]) for file in file_list]
    labels = np.array(labels)

    cal_acc(results, labels)
