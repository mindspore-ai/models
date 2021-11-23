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


from easydict import EasyDict as edict

vnet_cfg = edict(
    {
        'task': 'promise12',
        'fold': 0,
        # data setting
        'dirResult': 'results/infer',
        'dirPredictionImage': 'results/prediction',
        'normDir': False,
        'dstRes': [1, 1, 1.5],
        'VolSize': [128, 128, 64],
        # training setting
        'batch_size': 4,
        'epochs': 500,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'momentum': 0.99,
        'warmup_step': 120,
        'warmup_ratio': 0.3,
    }
)
