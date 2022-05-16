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
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

# config for squeezenet, imagenet
config_imagenet = ed({
    # Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
    "enable_modelarts": False,
    # Url for modelarts
    "data_url": "",
    "train_url": "",
    "checkpoint_url": "",
    # Path for local
    "run_distribute": False,
    "enable_profiling": False,
    "data_path": "/cache/data",
    "output_path": "/cache/train",
    "load_path": "/cache/checkpoint_path/",
    "device_num": 1,
    "device_id": 0,
    "device_target": "Ascend",
    "checkpoint_path": "./checkpoint/",
    "checkpoint_file_path": "suqeezenet_residual_imagenet-300_5004.ckpt",
    # Training options
    "net_name": "suqeezenet_residual",
    "dataset": "imagenet",
    "class_num": 1000,
    "batch_size": 32,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 7e-5,
    "epoch_size": 300,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 0,
    "lr_decay_mode": "cosine",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0,
    "lr_end": 0,
    "lr_max": 0.01,
    "pre_trained": "",
    #export
    "width": 227,
    "height": 227,
    "file_name": "squeezenet",
    "file_format": "MINDIR"
})

# config for squeezenet, cifar10
config_cifar = ed({
    # Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
    "enable_modelarts": False,
    # Url for modelarts
    "data_url": "",
    "train_url": "",
    "checkpoint_url": "",
    # Path for local
    "run_distribute": False,
    "enable_profiling": False,
    "data_path": "/cache/data",
    "output_path": "/cache/train",
    "load_path": "/cache/checkpoint_path/",
    "device_num": 1,
    "device_id": 0,
    "device_target": "Ascend",
    "checkpoint_path": "./checkpoint/",
    "checkpoint_file_path": "suqeezenet_residual_cifar10-150_195.ckpt",
    # Training options
    "net_name": "suqeezenet_residual",
    "dataset": "cifar10",
    "class_num": 10,
    "batch_size": 32,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 150,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 5,
    "lr_decay_mode": "linear",
    "lr_init": 0,
    "lr_end": 0,
    "lr_max": 0.01,
    "pre_trained": "",
    #export
    "width": 227,
    "height": 227,
    "file_name": "squeezenet",
    "file_format": "MINDIR"
})
