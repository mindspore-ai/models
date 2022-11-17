# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as edict

nasnet_a_mobile_config_gpu = edict({
    'random_seed': 1,
    'rank': 0,
    'group_size': 1,
    'work_nums': 8,
    'epoch_size': 600,
    'keep_checkpoint_max': 100,
    'ckpt_path': './checkpoint/',
    'is_save_on_master': 0,

    ### Dataset Config
    'train_batch_size': 32,
    'val_batch_size': 32,
    'image_size': 224,
    'num_classes': 1000,

    ### Loss Config
    'label_smooth_factor': 0.1,
    'aux_factor': 0.4,

    ### Learning Rate Config
    'lr_init': 0.04*8,
    'lr_decay_rate': 0.97,
    'num_epoch_per_decay': 2.4,

    ### Optimization Config
    'weight_decay': 0.00004,
    'momentum': 0.9,
    'opt_eps': 1.0,
    'rmsprop_decay': 0.9,
    "loss_scale": 1,

    ### Cutout Config
    'cutout': False,
    'cutout_length': 56,

})

nasnet_a_mobile_config_ascend = edict({
    'random_seed': 1,
    'rank': 0,
    'group_size': 1,
    'work_nums': 8,
    'epoch_size': 600,
    'keep_checkpoint_max': 50,
    'ckpt_path': './',
    'is_save_on_master': 0,

    ### Dataset Config
    'train_batch_size': 32,
    'val_batch_size': 32,
    'image_size': 224,
    'num_classes': 1000,

    ### Loss Config
    'label_smooth_factor': 0.1,
    'aux_factor': 0.4,

    ### Learning Rate Config
    'lr_init': 0.04*8,
    'lr_decay_rate': 0.97,
    'num_epoch_per_decay': 2.4,

    ### Optimization Config
    'weight_decay': 0.00004,
    'momentum': 0.9,
    'opt_eps': 1.0,
    'rmsprop_decay': 0.9,
    "loss_scale": 1,

    ### Cutout Config
    'cutout': True,
    'cutout_length': 56,

})
