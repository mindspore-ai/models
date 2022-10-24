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
"""get input data."""
import os
import argparse
import numpy as np
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/BSR/BSDS500/data/images/test',
                        help='evaling image path')
    parser.add_argument('--output_path', type=str, default='../data/input_bin',
                        help='output image path')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    path_list = os.listdir(opt.dataset_path)
    path_list.sort()
    if not os.path.exists(opt.dataset_path):
        os.makedirs(opt.dataset_path)

    for i, img_name in enumerate(path_list):
        print(img_name)
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_bin = img_name.split(".")[0] + '.bin'
            print(img_bin)
            img_path = os.path.join(opt.dataset_path, img_name)
            img_file = np.array(cv2.imread(img_path), dtype=np.float32)
            if img_file.shape[0] > img_file.shape[1]:
                img_file = np.rot90(img_file, 1).copy()
            img_file_origin = img_file.copy()
            print(img_file_origin)
            img_file -= np.array((104.00698793, 116.66876762, 122.67891434))
            print(img_file)
            img_file = np.transpose(img_file, (2, 0, 1))
            print(img_file.shape)

            print(opt.output_path+'/'+img_bin)

            img_file.tofile(opt.output_path+'/'+img_bin)
