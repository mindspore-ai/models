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

"""Configuration for training"""

from easydict import EasyDict as edict

facenet_cfg = edict({
    "rank": 0,
    "num_epochs": 240,
    "num_train_triplets": 3000000,
    "num_valid_triplets": 10000,
    "batch_size": 64,  #64
    "num_workers": 8,
    "learning_rate": 0.004,
    "margin": 0.5,
    "per_print_times": 1,
    "step_size": 50,
    "keep_checkpoint_max": 10,
    "lr_epochs": '30,60,90,120,150,180,210,240',
    "lr_gamma": 0.1,
    "T_max": 200,
    "warmup_epochs": 0,
    "device_target": "Ascend"
})
