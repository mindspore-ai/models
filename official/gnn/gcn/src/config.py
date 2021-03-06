# Copyright 2020 Huawei Technologies Co., Ltd
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


class ConfigGCN():
    """Configuration for GCN"""
    learning_rate = 0.01
    epochs = 200
    hidden1 = 16
    dropout = 0.5
    weight_decay = 5e-4
    early_stopping = 50
    save_ckpt_steps = 549
    keep_ckpt_max = 10
    ckpt_dir = './ckpt'
    best_ckpt_dir = './best_ckpt'
    best_ckpt_name = 'best.ckpt'
    eval_start_epoch = 100
    save_best_ckpt = True
    eval_interval = 1
