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


export DEVICE_NUM=8
export CONFIG_PATH="src/Sony_config.yaml"

echo "Training on GPUs"
echo "Using $DEVICE_NUM gpus"
echo "Config path is "$CONFIG_PATH

mpirun -n $DEVICE_NUM --allow-run-as-root python train_sony.py --device_target GPU --run_distribute=True \
 --device_num $DEVICE_NUM --config $CONFIG_PATH &> train.log &
