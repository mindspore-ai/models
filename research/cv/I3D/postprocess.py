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
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-path', type=str, default=None)
    args = parser.parse_args()

    bs = args.batch_size
    label_path = args.label_path
    datapath = os.path.abspath(os.path.dirname(
        __file__)) + "/scripts/result_Files"
    file_num = len(os.listdir(datapath))

    acc_sum, sample_num = 0, 0
    label_list = np.load(label_path, allow_pickle=True)
    for i in range(file_num):
        clippath = os.path.abspath(os.path.dirname(
            __file__)) + os.path.join("/scripts/result_Files",
                                      str(args.dataset) + '_bs' + str(args.mode) + '_' + str(i + 1) + '_0' + '.bin')
        predictions = np.fromfile(clippath, np.float32)
        predictions = predictions.reshape(bs, args.num_classes)
        rows, _ = predictions.shape
        label = label_list[i]
        print('label shape:', label.shape)
        for j in range(rows):
            preds = np.argmax(predictions[j])
            target = label[j]
            acc = (preds == target)
            acc_sum += acc
        batch_size = label.shape[0]
        sample_num += batch_size
        accuracy_top = acc_sum / sample_num
        print('result:{:.3f}%'.format(accuracy_top * 100))

    accuracy_top1 = acc_sum / sample_num
    print('eval result: top_1 {:.3f}%'.format(accuracy_top1 * 100))


if __name__ == '__main__':
    cal_acc()
