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
"""Configuration file."""

import importlib

experiment_cfg = {
    'device': 'GPU',
    'model_config': 'litehrnet_18_coco_256x192',
    'learning_rate': 5e-5,
    'loss_scale': 2 ** 16,
    'weight_decay': 1e-6,
    'experiment_tag': 'lhrnet18_256_coco_default',
    'checkpoint_path': None,
    'start_epoch': 0,
    'checkpoint_interval': 5,
    'random_seed': None
}

model_cfg = importlib.import_module('configs.' + experiment_cfg['model_config'])
