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
config = {
    "dataset_path": "./dataset/",
    "backbone_pretrained": "./src/model/res2net_pretrained.ckpt",
    "dataset_train": "PASCAL_SBD",
    "datasets_val": ["GrabCut", "Berkeley"],
    "epochs": 33,
    "train_only_epochs": 32,
    "val_robot_interval": 1,
    "lr": 0.007,
    "batch_size": 8,
    "max_num": 0,
    "size": (384, 384),
    "device": "Ascend",
    "num_workers": 4,
    "itis_pro": 0.7,
    "max_point_num": 20,
    "record_point_num": 5,
    "pred_tsh": 0.5,
    "miou_target": [0.90, 0.90],
}
