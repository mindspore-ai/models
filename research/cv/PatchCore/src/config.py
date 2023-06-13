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
"""config"""
from yacs.config import CfgNode as CN

_C = CN()

# Pre/Post -process
_C.mean_dft = [-0.485, -0.456, -0.406]
_C.std_dft = [0.229, 0.224, 0.255]
# Computed from random subset of ImageNet training images
_C.mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255]
_C.std = [1 / 0.229, 1 / 0.224, 1 / 0.255]

# Train/eval
_C.train_url = ""
_C.data_url = ""
_C.isModelArts = False

_C.category = "screw"
_C.coreset_sampling_ratio = 0.01
_C.num_epochs = 1
_C.device_id = 0
_C.dataset_path = ""
_C.pre_ckpt_path = ""  # Pretrain checkpoint file path

_C.platform = "Ascend"

cfg = _C


def merge_from_cli_list(args: list):
    print(args)
    stripped_args = []
    for arg in args:
        if arg[:2] == "--":
            stripped_args.append(arg[2:])
        elif arg[:1] == "-":
            stripped_args.append(arg[1:])
        else:
            stripped_args.append(arg)
    print(stripped_args)
    cfg.merge_from_list(stripped_args)
