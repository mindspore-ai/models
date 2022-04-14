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
from easydict import EasyDict as ed

config_train = ed({
    "net_name": "resnest50",
    "root": "/data1/datasets/imagenet/",
    "base_size": 224,
    "crop_size": 224,
    "num_classes": 1000,
    "label_smoothing": 0.1,
    "batch_size": 64,
    "test_batch_size": 64,
    "last_gamma": True,
    "final_drop": 1.0,
    "epochs": 270,
    "start_epoch": 0,
    "num_workers": 50,

    "lr": 0.025,
    "steps_per_epoch": 1,
    "lr_epochs": '30,60,90,120,150,180,210,240,270',
    "lr_gamma": 0.1,
    "max_epoch": 270,
    "eta_min": 0,
    "T_max": 270,
    "lr_scheduler": "cosine_annealing",
    "is_dynamic_loss_scale": 1,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "disable_bn_wd": True,
    "warmup_epochs": 1
})
