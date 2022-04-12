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

if [ $# != 2 ]
then
    echo "===================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_distribute_train_gpu.sh [DATASET_PATH] [OUTPUT_CKPT_DIR]"
    echo "for example: bash scripts/run_distribute_train_gpu.sh /path/to/dataset/root /save/checkpoint/directory"
    echo "===================================================================================================="
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD/$1"
  fi
}

DATASET_PATH=$(get_real_path "$1")
OUTPUT_CKPT_DIR=$(get_real_path "$2")

if [ ! -d "$DATASET_PATH" ] ; then
    echo "Cannot find the specified dataset directory: $DATASET_PATH"
    exit 1
fi

if [ -d logs ]
then
  rm -r logs
fi

mkdir logs

mpirun --output-filename logs\
       -np 8 --allow-run-as-root \
       python ./train.py \
       --dataset_root "$DATASET_PATH" \
       --ckpt_save_directory "$OUTPUT_CKPT_DIR" \
       --is_train_distributed True \
       --learning_rate 0.0005 \
       > ./logs/train.log 2>&1 &
