#!/bin/bash
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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: sh run_standalone_eval_cpu.sh [cifar10] [SIMCLR_MODEL_PATH] [TRAIN_DATASET_PATH] [EVAL_DATASET_PATH]"
exit 1
else

script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
export DATASET_NAME=$1
export SIMCLR_MODEL_PATH=$2
export TRAIN_DATASET_PATH=$3
export EVAL_DATASET_PATH=$4


python ${self_path}/../linear_eval.py --dataset_name=$DATASET_NAME \
               --encoder_checkpoint_path=$SIMCLR_MODEL_PATH \
               --train_dataset_path=$TRAIN_DATASET_PATH \
               --eval_dataset_path=$EVAL_DATASET_PATH \
               --device_target="CPU" \
               --run_distribute=False --run_cloudbrain=False > eval_log 2>&1 &
fi
