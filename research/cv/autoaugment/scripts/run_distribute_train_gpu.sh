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

rm -rf logs
mkdir ./logs

mpirun -n 8 --allow-run-as-root python train.py \
        --dataset svhn \
        --dataset_path $1 \
        --run_distribute=True \
        --device_target=GPU \
        --lr_init=0.08 \
        --epoch_size=50 \
    > ./logs/distributed_train.log 2>&1 &
