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
"""preprocess dataset"""
import os
import argparse
from PIL import Image
from scipy.io import loadmat


def _parser_args():
    parser = argparse.ArgumentParser('dataset list generator')
    parser.add_argument("--data_dir", type=str, default='', help="VOC2012 data dir")
    return parser.parse_args()


def _get_data_list(data_list_file):
    with open(data_list_file, 'r') as f:
        return f.readlines()


def _mat_to_arr(mat_path):
    data = loadmat(mat_path)['GTcls']
    arr = data[0, 0][1]
    return arr


def main():
    args = _parser_args()
    data_dir = args.data_dir
    cls_path = os.path.join(data_dir, 'cls')
    cls_png_path = os.path.join(data_dir, 'cls_png')
    if not os.path.exists(cls_png_path):
        os.mkdir(cls_png_path)
    mat_list = os.listdir(cls_path)
    print('Start generating png.')
    print("It takes a little time. Don't quit!")
    i = 0
    for mat in mat_list:
        mat_path = os.path.join(cls_path, mat)
        arr = _mat_to_arr(mat_path)
        png_path = os.path.join(cls_png_path, mat.replace('mat', 'png'))
        ann_im = Image.fromarray(arr)
        ann_im.save(png_path)
        i += 1
    print(f"Generate {i} png to data_dir/cls_png.")

    train_txt = os.path.join(data_dir, 'train.txt')
    train_list_txt = os.path.join(data_dir, 'train_list.txt')
    val_txt = os.path.join(data_dir, 'val.txt')
    val_list_txt = os.path.join(data_dir, 'val_list.txt')

    train_data_lst = _get_data_list(train_txt)
    with open(train_list_txt, 'w') as f:
        for line in train_data_lst:
            line = line.strip()
            img_ = os.path.join('img', line + '.jpg')
            anno_ = os.path.join('cls_png', line + '.png')
            f.write(f'{img_} {anno_}\n')
    print('Generate train_list to data_dir.')

    val_data_lst = _get_data_list(val_txt)
    with open(val_list_txt, 'w') as f:
        for line in val_data_lst:
            line = line.strip()
            img_ = os.path.join('img', line + '.jpg')
            anno_ = os.path.join('cls_png', line + '.png')
            f.write(f'{img_} {anno_}\n')
    print('Generate val_list to data_dir.')
    print('Finish.')


if __name__ == "__main__":
    main()
