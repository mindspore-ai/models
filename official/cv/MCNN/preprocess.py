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
"""pre process for 310 inference"""
import os
import argparse
import cv2
import numpy as np

batch_size = 1
parser = argparse.ArgumentParser(description="mcnn preprocess data")
parser.add_argument("--dataset_path", type=str, default="./test_data/images/", help="dataset path.")
parser.add_argument("--output_path", type=str, default="./test_data/preprocess_data/", help="output path.")
args = parser.parse_args()


def save_mnist_to_jpg(data_path, output_path):
    data_files = [filename for filename in os.listdir(data_path) \
                  if os.path.isfile(os.path.join(data_path, filename))]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for fname in data_files:
        img = cv2.imread(os.path.join(data_path, fname), 0)
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht // 4) * 4
        wd_1 = (wd // 4) * 4
        img = cv2.resize(img, (wd_1, ht_1))
        hang = (1024 - ht_1) // 2
        lie = (1024 - wd_1) // 2
        img = np.pad(img, ((hang, hang), (lie, lie)), 'constant')
        img.tofile(os.path.join(output_path, fname+'.bin'))


if __name__ == '__main__':
    save_mnist_to_jpg(args.dataset_path, args.output_path)
