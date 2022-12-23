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

DEVICE=CPU

echo "export mindir models after distillation with ATS"
python export.py --device $DEVICE --device_id 0 --n_classes 100 --t_net_name ResNet14 --net_name VGG8 --ckpt_dir ./ckpts --ckpt_name cifar100-ResNet14-VGG8-ATS.ckpt
python export.py --device $DEVICE --device_id 0 --n_classes 100 --t_net_name ResNet110 --net_name VGG8 --ckpt_dir ./ckpts --ckpt_name cifar100-ResNet110-VGG8-ATS.ckpt
