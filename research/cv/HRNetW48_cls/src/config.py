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
"""Configuration for HRNet-Classification."""
import json
from easydict import EasyDict as ed


config_hrnetw48_cls = ed({
    "train_url": None,
    "train_path": None,
    "data_url": None,
    "data_path": None,
    "checkpoint_url": None,
    "checkpoint_path": None,
    "eval_data_url": None,
    "eval_data_path": None,
    "eval_interval": 10,
    "modelarts": False,
    "run_distribute": False,
    "device_target": "Ascend",
    "begin_epoch": 0,
    "end_epoch": 120,
    "total_epoch": 120,
    "dataset": "imagenet",
    "num_classes": 1000,
    "batchsize": 16,
    "input_size": 224,
    "lr_scheme": "linear",
    "lr": 0.01,
    "lr_init": 0.0001,
    "lr_end": 0.00001,
    "warmup_epochs": 2,
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "conv_init": "TruncatedNormal",
    "dense_init": "RandomNormal",
    "optimizer": "rmsprop",
    "loss_scale": 1024,
    "opt_momentum": 0.9,
    "wd": 0.00001,
    "eps": 0.001,
    "save_ckpt": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 10,
    "model": ed({
        "name": "cls_hrnet_w48",
        "extra": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [4],
                "NUM_CHANNELS": [64],
                "FUSE_METHOD": "SUM"
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [48, 96],
                "FUSE_METHOD": "SUM"
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192],
                "FUSE_METHOD": "SUM"
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192, 384],
                "FUSE_METHOD": "SUM"
            }
        },
    }),
})


def show_config(cfg):
    """Show configuration."""
    split_line_up = "==================================================\n"
    split_line_bt = "\n=================================================="
    print(split_line_up,
          json.dumps(cfg, ensure_ascii=False, indent=2),
          split_line_bt)


def organize_configuration(cfg, args):
    """Add parameters from command-line into configuration."""
    args_dict = vars(args)
    for item in args_dict.items():
        cfg[item[0]] = item[1]
