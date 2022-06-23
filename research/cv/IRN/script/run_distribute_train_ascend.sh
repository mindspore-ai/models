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

if [ $# != 3 ]; then
    echo "Usage: 
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [SCALE] [DATASET_PATH]"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "error: RANK_TABLE_FILE:$1 does not exist"
    exit 1
fi

if [ ! -d $3 ]; then
    echo "error: DATASET_PATH:$3 does not exist"
    exit 1
fi

export RANK_TABLE_FILE=$1
export DEVICE_NUM=8
export RANK_SIZE=8

PID_LIST=()
for ((i=0; i<${RANK_SIZE}; i++)); do
    export DEVICE_ID=${i}
    export RANK_ID=$((i))
    echo "Start distributed training for rank ${RANK_ID}, device ${DEVICE_ID}"
    nohup python ../train.py \
        --scale $2 \
        --dataset_GT_path $3 \
        --device_target Ascend \
        --run_distribute=True \
        > ../train-${i}.log 2>&1 & 
    pid=$!
    PID_LIST+=("${pid}")
done

RUN_BACKGROUND=1
if (( RUN_BACKGROUND == 0 )); then
  echo "Waiting for all processes to exit..."
  for pid in ${PID_LIST[*]}; do
      wait ${pid}
      echo "Process ${pid} exited"
  done
fi