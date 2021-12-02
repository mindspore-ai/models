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
"""preprocess"""
from __future__ import print_function
import argparse
import os
import numpy as np
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process file')
    parser.add_argument('--val_dataset_folder', type=str, default='/home/dataset/widerface/val',
                        help='val dataset folder.')
    args = parser.parse_args()

    # testing dataset
    test_dataset = []
    with open(os.path.join(args.val_dataset_folder, 'val_img_list.txt'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        test_dataset.append(line.rstrip())

    # transform data to bin_file
    img_path = "./bin_file"
    if os.path.exists(img_path):
        os.system('rm -rf ' + img_path)
    os.makedirs(img_path)
    h_max, w_max = 1024, 1024
    for i, img_name in enumerate(test_dataset):
        image_path = os.path.join(args.val_dataset_folder, 'images', img_name)

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        img = cv2.resize(img, (1024, 1024))

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)  # [1, c, h, w]

        # save bin file
        file_name = "widerface_test" + "_" + str(i) + ".bin"
        file_path = os.path.join(img_path, file_name)
        img.tofile(file_path)
