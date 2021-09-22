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
network config setting, will be used in train.py
"""

from easydict import EasyDict as Edict

zfnet_cfg = Edict({
    'num_classes': 1000,
    'momentum': 0.9,
    'epoch_size': 150,
    'batch_size': 256,
    'image_height': 224,
    'image_width': 224,
    'warmup_epochs': 5,
    'iteration_max': 150,
    "lr_init": 0.035,
    "lr_min": 0.0,
    # opt
    'weight_decay': 0.0001,
    'loss_scale': 1024,
    # lr
    'is_dynamic_loss_scale': 0,
})

sppnet_single_cfg = Edict({
    'num_classes': 1000,
    'momentum': 0.9,
    'epoch_size': 160,
    'batch_size': 256,
    'image_height': 224,
    'image_width': 224,
    'warmup_epochs': 0,
    'iteration_max': 150,
    "lr_init": 0.01,
    "lr_min": 0.0,
    # opt
    'weight_decay': 0.0001,
    'loss_scale': 1024,
    # lr
    'is_dynamic_loss_scale': 0,
})

sppnet_mult_cfg = Edict({
    'num_classes': 1000,
    'momentum': 0.9,
    'epoch_size': 160,
    'batch_size': 128,
    'image_height': 224,
    'image_width': 224,
    'warmup_epochs': 2,
    'iteration_max': 150,
    "lr_init": 0.01,
    "lr_min": 0.0,
    # opt
    'weight_decay': 0.0001,
    'loss_scale': 1024,
    # lr
    'is_dynamic_loss_scale': 0,
})
