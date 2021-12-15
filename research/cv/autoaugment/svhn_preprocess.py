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

"""
Extract SVHN Dataset (Format 2) from .mat files (http://ufldl.stanford.edu/housenumbers/)
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

parser = argparse.ArgumentParser(description="preprocess svhn dataset")
parser.add_argument("--result_dir", type=str, required=True, help="result files path.")
parser.add_argument("--mat_path", type=str, required=True, help="Path to folder with .mat files")
args = parser.parse_args()

train_names = ['train_32x32.mat', 'extra_32x32.mat']
test_name = 'test_32x32.mat'

Path(args.result_dir, 'train').mkdir(parents=True, exist_ok=True)
Path(args.result_dir, 'test').mkdir(parents=True, exist_ok=True)


def read_and_save_imgs(mat, dir_name):

    """
    Reads .mat files from folder sequentially and saves images to folders by their labels.
    """

    my_sub_dir_count = defaultdict(int)
    loaded_mat = sio.loadmat(str(mat))
    data = loaded_mat['X']
    labels = loaded_mat['y'].astype(np.int64).squeeze()
    for i in range(data.shape[3]):
        img = Image.fromarray(data[:, :, :, i])
        label = int(labels[i])
        sub_dir_path = Path(args.result_dir, dir_name, str(label))
        if not sub_dir_path.exists():
            sub_dir_path.mkdir(parents=True, exist_ok=True)
        my_sub_dir_count[label] += 1
        file_index = my_sub_dir_count[label]
        file_path = sub_dir_path / f'{file_index}.jpeg'
        img.save(file_path)


def preprocess_svhn():
    for mat in train_names:
        print(f"convert {mat} file")
        read_and_save_imgs(Path(args.mat_path, mat), 'train')
    print(f"convert {test_name} file")
    read_and_save_imgs(Path(args.mat_path, test_name), 'test')
    print('success')


if __name__ == '__main__':
    preprocess_svhn()
