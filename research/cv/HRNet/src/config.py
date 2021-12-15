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
"""Configuration."""
import json
from easydict import EasyDict as ed


# Configuration for training and inference
task_config = ed({
    "lr": 1e-4,
    "lr_min": 1e-7,
    "lr_scheme": "cos",
    "optimizer": "adam",
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "total_epoch": 600,
    "batchsize": 4,
    "loss_scale": 1024,
    "wd": 1e-5
})


# Configuration for model definition
hrnetw48_config = ed({
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
})


def show_config(cfg):
    """Show configuration."""
    split_line_up = "==================================================\n"
    split_line_bt = "\n=================================================="
    print(split_line_up,
          json.dumps(cfg, ensure_ascii=False, indent=2),
          split_line_bt, flush=True)
