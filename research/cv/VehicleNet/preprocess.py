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
"""preprocess
"""
import os
import argparse
import numpy as np

def load_bin(dataset_type, bin_path, img_path, image_size):
    '''load evalset of .bin
    '''
    files = os.listdir(bin_path)
    files.sort()

    num = 0
    for file in files:
        num = num + 1

    shape = (num * 2, 1, 3, image_size, image_size)
    data_list = np.zeros(shape, np.float32)
    idx = 0
    for file in files:
        file_name = os.path.join(bin_path, file)
        f = open(file_name, mode='rb')
        img = np.fromfile(f, dtype=np.float32).reshape(1, 3, image_size, image_size)
        data_list[idx, :] = img
        for i in range(img.shape[0]):
            img[i] = np.transpose(np.fliplr(np.transpose(img[i], (0, 2, 1))), (0, 2, 1))
        data_list[idx + 1, :] = img
        idx = idx + 2

    j = 0
    for data in data_list:
        file_path = os.path.join(img_path, 'VehicleNet_' + dataset_type + '_bs1' + '_' + str(format(j, '08d')) + '.bin')
        j = j + 1
        data.tofile(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do preprocess')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument("--test_bin_path", type=str, help="")
    parser.add_argument("--query_bin_path", type=str, help="")
    parser.add_argument("--test_path", type=str, help="")
    parser.add_argument("--query_path", type=str, help="")
    args = parser.parse_args()

    img_size = 384
    load_bin('test', args.test_bin_path, args.test_path, img_size)
    load_bin('query', args.query_bin_path, args.query_path, img_size)
