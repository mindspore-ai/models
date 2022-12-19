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

echo "Train teacher network ResNet14 on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_classify.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --net ResNet --n_layer 14 --net_name ResNet14 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name teacher.log --ckpt_name cifar100-ResNet14.ckpt

echo "Train teacher network ResNet110 on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_classify.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --net ResNet --n_layer 110 --net_name ResNet110 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name teacher.log --ckpt_name cifar100-ResNet110.ckpt
