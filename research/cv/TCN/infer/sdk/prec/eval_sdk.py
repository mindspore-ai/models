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
""" eval_sdk.py """
import os
import numpy as np


def read_file_list(input_f):
    """
    :param infer file content:
        1.bin 0
        2.bin 2
        ...
    :return image path list, label list
    """
    image_f = []
    labels_l = []
    if not os.path.exists(input_f):
        print('input file does not exists.')
    with open(input_f, "r") as fs:
        for line in fs.readlines():
            line = line.strip('\n').split(',')
            file_name = line[0]
            label = int(line[1])
            image_f.append(file_name)
            labels_l.append(label)
    return image_f, labels_l

images_txt_path = "../data/mnist_infer_data/mnist_bs_1_label.txt"

file_list, label_list = read_file_list(images_txt_path)
img_size = len(file_list)
labels = np.array(label_list)

results = np.loadtxt('result/infer_results.txt')
acc = (results == labels).sum() / img_size
print('total acc:', acc)
