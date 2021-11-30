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
"""Preprocess for DeepID"""
import os
import time
import argparse

from src.dataset import dataloader

parser = argparse.ArgumentParser(description='DeepID_preprocess')

parser.add_argument('--data_url', type=str, default='data/', help='Dataset path')
parser.add_argument('--save_url', type=str, default='../bin_data', help='Dataset path')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
parser.add_argument('--mode', type=str, default='valid', help='dataset mode')

if __name__ == '__main__':
    args_opt = parser.parse_args()

    valid_dataset, valid_dataset_length = dataloader(args_opt.data_url, epoch=1,
                                                     mode=args_opt.mode, batch_size=args_opt.batch_size)

    valid_dataset_iter = valid_dataset.create_dict_iterator()
    print('Valid dataset length:', valid_dataset_length)

    img_path = os.path.join(args_opt.save_url, "img_data")
    label_path = os.path.join(args_opt.save_url, "label")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    for idx, data in enumerate(valid_dataset_iter):
        step_begin_time = time.time()
        img_valid = data['image']
        label_valid = data['label']
        file_name = "sop_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        label_file_path = os.path.join(label_path, file_name)
        img_valid.asnumpy().tofile(img_file_path)
        label_valid.asnumpy().tofile(label_file_path)
        print('Finish processing img', idx, "saving as", file_name)

    print("=" * 20, "export bin files finished", "=" * 20)
