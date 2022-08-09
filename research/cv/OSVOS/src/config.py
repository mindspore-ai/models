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
network config setting.
"""
from easydict import EasyDict as edict


osvos_cfg = edict(
    {
        'task': 'OSVOS',
        'dirResult': './results',

        # data path,
        'models_save_path': './models',

        # train_parent setting
        'tp_batch_size': 12,
        'tp_lr': 5e-5,
        'tp_wd': 0.0002,
        'tp_epoch_size': 240,

        # train_online setting
        'to_batch_size': 1,
        'to_epoch_size': 10000,
        'to_lr': 0.000005,
        'to_wd': 0.0002,
    }
)
