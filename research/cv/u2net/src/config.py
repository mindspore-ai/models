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
"""config of distribute training and standanlone training"""
from easydict import EasyDict as edict

single_cfg = edict({
    "lr": 1e-3,
    "batch_size": 12,
    "max_epoch": 1000,
    "keep_checkpoint_max": 3,
    "weight_decay": 0,
    "eps": 1e-8,
})
run_distribute_cfg = edict({
    "lr": 4e-3,
    "batch_size": 16,
    "max_epoch": 1500,
    "keep_checkpoint_max": 3,
    "weight_decay": 0,
    "eps": 1e-8,
})
single_cfg_GPU = edict({
    "lr": 1e-3,
    "batch_size": 12,
    "max_epoch": 1200,
    "keep_checkpoint_max": 3,
    "weight_decay": 0,
    "eps": 1e-8,
})
run_distribute_cfg_GPU = edict({
    "lr": 4e-3,
    "batch_size": 16,
    "max_epoch": 2150,
    "keep_checkpoint_max": 3,
    "weight_decay": 0,
    "eps": 1e-8,
})
