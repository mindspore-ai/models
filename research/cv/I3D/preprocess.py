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
Preprocessing of I3D model datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import src.data_factory as data_factory
from src.transforms.spatial_transforms import Compose, RandomHorizontalFlip, RandomCrop, CenterCrop
from src.transforms.target_transforms import ClassLabel
from src.transforms.temporal_transforms import TemporalRandomCrop
from src.utils import print_config, write_config, prepare_output_dirs
from config import parse_opts


def run():
    config = parse_opts()
    if config.dataset == 'ucf101':
        config.finetune_num_classes = 101

    if config.distributed:
        config.save_dir = './output_distribute/'
    config = prepare_output_dirs(config)
    print_config(config)
    write_config(config, os.path.join(config.save_dir, 'config.json'))

    train_transforms = {'spatial': Compose([RandomCrop(config.spatial_size), RandomHorizontalFlip()]),
                        'temporal': TemporalRandomCrop(config.train_sample_duration),
                        'target': ClassLabel()}
    validation_transforms = {'spatial': Compose([CenterCrop(config.spatial_size)]),
                             'temporal': TemporalRandomCrop(config.test_sample_duration),
                             'target': ClassLabel()}

    dataset = data_factory.get_dataset(config, train_transforms, validation_transforms)
    data_path = os.path.abspath(os.path.dirname(
        __file__)) + os.path.join("/preprocess_Result/data", str(config.dataset), str(config.mode))
    label_path = os.path.abspath(os.path.dirname(
        __file__)) + os.path.join("/preprocess_Result/label", str(config.dataset), str(config.mode))
    label_list = []
    print(data_path)
    file_path = os.path.join(label_path, "label_bs" + str(config.mode) + str(config.batch_size) + ".npy")
    if os.path.exists(data_path):
        print("=====================flag=================")
        os.system('rm -rf ' + data_path)
    os.makedirs(data_path)
    if os.path.exists(label_path):
        print("=====================flag=================")
        os.system('rm -rf ' + label_path)
    os.makedirs(label_path)

    i = 0
    for data in dataset['train'].create_dict_iterator(output_numpy=True):
        i = i + 1
        clip = data['clip']
        file_name = str(config.dataset) + '_bs' + \
                    str(config.mode) + '_' + str(i) + '.bin'
        file_data_path = os.path.join(data_path, file_name)
        print('clip file ', i, 'writing')
        clip.tofile(file_data_path)
        print('clip preprocess down.')
        label = data['target']
        label_list.append(label)
    np.save(file_path, label_list)
    print('Finished training.')


if __name__ == '__main__':
    run()
