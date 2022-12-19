#!/bin/bash
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

EPOCHES=2

echo "(epoches=$EPOCHES for demo, change to 240 for training)"

echo "Train student network VGG8 on CIFAR-100 without KD (epoches=2 for demo, change to 240 for training)"
python run_classify.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --net VGG --n_layer 8 --net_name VGG8 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-nokd.log --ckpt_name cifar100-VGG8.ckpt
