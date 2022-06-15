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
if [ $# != 3 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_squad_distribute_train_gpu.sh [DATA] [MODEL_FILE] [TRAIN_BATCH_SIZE]"
    exit 1
fi


get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

DATA=$(get_real_path "$1")
MODEL_FILE=$(get_real_path "$2")
TRAIN_BATCH_SIZE=$3

if [ ! -d "$DATA" ]
then
  echo "The specified dataset root is not a directory: \"$DATA\"."
  exit 1
fi

if [ ! -d "$MODEL_FILE" ]
then
  echo "The specified model file  root is not a directory: \"$MODEL_FILE\"."
  echo "[MODEL_FILE] is  the path to the folder that contains the unpacked luke_large_500k.tar.gz files."
  exit 1
fi

rm -rf logs
mkdir ./logs
mpirun -n 8 --allow-run-as-root python run_squad_train.py --data="$DATA"  --model_file="$MODEL_FILE" \
   --train_batch_size="$TRAIN_BATCH_SIZE" --num_train_epochs=2 --learning_rate=12e-6 --distribute=True \
        > ./logs/distributed_train_squad.log 2>&1 &
