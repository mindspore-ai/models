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

tsn_flow = edict({
    'learning_rate': 0.005,
    'epochs': 340,
    'lr_steps': 70,
    'gamma': 0.1,
    'dropout': 0.3,
    'num_segments': 3,
})

tsn_rgb = edict({
    'learning_rate': 0.003,
    'epochs': 80,
    'lr_steps': 30,
    'gamma': 0.1,
    'dropout': 0.2,
})

tsn_rgb_diff = edict({
    'learning_rate': 0.003,
    'epochs': 180,
    'lr_steps': 80,
    'gamma': 0.07,
    'dropout': 0.2,
})
