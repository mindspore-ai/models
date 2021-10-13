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
"""
##############preprocess#################
"""
import os
import pickle
import argparse
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt


def load_bin(path, image_size):
    '''load evalset of .bin
    '''
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as _:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for _ in [0, 1]:
        data = np.zeros(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = plt.imread(BytesIO(_bin), "jpg")
        img = np.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2)
            data_list[flip][idx][:] = img
    print("data_list:", len(data_list))
    return data_list, issame_list


def test(data_set, batch_size, label_dir):
    '''test
    '''
    print('testing preprocess')
    data_list = data_set[0]
    issame_list = data_set[1]
    np.save(os.path.join(label_dir, "issame_list.npy"), issame_list)
    i = 0
    print(len(data_list))
    print(len(issame_list))
    for data in data_list:
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            _data = data[bb - batch_size: bb]

            img = ((_data / 255) - 0.5) / 0.5
            ba = bb
            img = img.astype(np.float32)
            file_path = os.path.join(img_path, "lfw" + '_' + str(i) + '.bin')
            i = i + 1
            img.tofile(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do preprocess')
    parser.add_argument("--result_dir", type=str, help="")
    parser.add_argument("--label_dir", type=str, help="")
    parser.add_argument('--batch_size', default=64, type=int, help='')
    parser.add_argument("--dataset_path", type=str, help="")
    args = parser.parse_args()
    img_path = os.path.join(args.result_dir)
    img_size = [112, 112]
    dataset = load_bin(args.dataset_path, img_size)
    test(dataset, args.batch_size, args.label_dir)
    print("="*20, "export bin files finished", "="*20)
