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
import argparse
import numpy as np
from mindspore.nn import Top1CategoricalAccuracy, Top5CategoricalAccuracy

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--dataset", type=str, required=True, help="dataset type.")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--label_path", type=str, required=True, help="image file path.")
args_opt = parser.parse_args()

if args_opt.dataset == "cifar10":
    from src.config import config1 as config
elif args_opt.dataset == "cifar100":
    from src.config import config2 as config
else:
    raise ValueError("dataset is not support.")

def cal_acc_cifar(result_path, label_path):
    '''calculate cifar accuracy'''
    top1_acc = Top1CategoricalAccuracy()
    top5_acc = Top5CategoricalAccuracy()
    result_shape = (config.batch_size, config.class_num)

    file_num = len(os.listdir(result_path))
    label_list = np.load(label_path)
    for i in range(file_num):
        f_name = args_opt.dataset + "_bs" + str(config.batch_size) + "_" + str(i) + "_0.bin"
        full_file_path = os.path.join(result_path, f_name)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape)
            gt_classes = label_list[i]

            top1_acc.update(result, gt_classes)
            top5_acc.update(result, gt_classes)
    print("top1 acc: ", top1_acc.eval())
    print("top5 acc: ", top5_acc.eval())


if __name__ == '__main__':
    cal_acc_cifar(args_opt.result_path, args_opt.label_path)
