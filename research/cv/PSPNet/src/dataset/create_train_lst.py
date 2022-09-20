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
"""generate train_lst.txt"""
import os
import argparse


def _parser_args():
    parser = argparse.ArgumentParser('dataset list generator')
    parser.add_argument("--data_dir", type=str, default='', help="VOC2012 data dir")
    return parser.parse_args()


def _get_data_list(data_list_file):
    with open(data_list_file, 'r') as f:
        return f.readlines()


def main():
    args = _parser_args()
    data_dir = args.data_dir
    voc_train_lst_txt = os.path.join(data_dir, 'voc_train_lst.txt')
    train_lst_txt = os.path.join(data_dir, 'train_lst.txt')

    voc_train_data_lst = _get_data_list(voc_train_lst_txt)
    with open(train_lst_txt, 'w') as f:
        for line in voc_train_data_lst:
            img_, anno_ = (os.path.join('VOCdevkit/VOC2012', i.strip()) for i in line.split())
            f.write(f'{img_} {anno_}\n')
    print('generating voc train list success.')


if __name__ == "__main__":
    main()
