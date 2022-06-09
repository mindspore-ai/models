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
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_distribute_train_gpu.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH]"
  exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

DATA_PATH=$(get_real_path "$1")
OUTPUT_PATH=$(get_real_path "$2")

if [ !  -d "$DATA_PATH" ]; then
  echo "Can not find the folder at the given path [DATASET_PATH]: $DATA_PATH"
  exit
fi

if [ !  -d "$OUTPUT_PATH" ]; then
  mkdir "$OUTPUT_PATH"
fi

mpirun -n 8 --allow-run-as-root \
    python train.py \
    --device_target="GPU" \
    --data_path="$DATA_PATH" \
    --train_path="$OUTPUT_PATH" \
    --run_distribute=True \
    --batchsize=64 \
    --lr=0.04 \
    --lr_init=0.0004 \
    --lr_end=0.00004 \
    --sink_mode \
    > "$OUTPUT_PATH"/distribute_train.log 2>&1 &
