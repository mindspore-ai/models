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
Functional Cells to be used.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time

import mindspore.nn as nn
import simplejson


def print_config(config):
    print('#' * 60)
    print('Training configuration:')
    for k, v in vars(config).items():
        print('  {:>20} {}'.format(k, v))
    print('#' * 60)


def write_config(config, json_path):
    with open(json_path, 'w') as f:
        f.write(simplejson.dumps(vars(config), indent=4, sort_keys=True))


def output_subdir(config):
    prefix = time.strftime('%Y%m%d_%H%M')
    if config.distributed:
        subdir = "{}_{}_device{}".format(config.dataset, config.mode, config.device_id)
    else:
        subdir = "{}_{}_{}_device{}".format(prefix, config.dataset, config.mode, config.device_id)
    return os.path.join(config.save_dir, subdir)


def prepare_output_dirs(config):
    config.save_dir = output_subdir(config)
    config.checkpoint_dir = os.path.join(config.save_dir, 'checkpoints')

    if os.path.exists(config.save_dir):
        shutil.rmtree(config.save_dir)

    if config.openI:
        os.makedirs(config.save_dir)
    else:
        os.mkdir(config.save_dir)
    os.mkdir(config.checkpoint_dir)
    return config


def get_optimizer(config, params, lr):
    if config.optimizer == 'SGD':
        return nn.SGD(params, lr, config.momentum, weight_decay=config.weight_decay)
    if config.optimizer == 'rmsprop':
        return nn.RMSProp(params, lr, weight_decay=config.weight_decay)
    if config.optimizer == 'adam':
        return nn.Adam(params, lr, weight_decay=config.weight_decay)
    raise ValueError('Chosen optimizer is not supported, please choose from (SGD | adam | rmsprop)')


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value
