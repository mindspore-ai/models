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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash scripts/run_distributed_train_gpu.sh [RANK_SIZE] [DATASET_PATH]"
    echo "or bash scripts/run_distributed_train_gpu.sh [RANK_SIZE] [DATASET_PATH] [PRE_TRAINED_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

export DEVICE_NUM=$1
export RANK_SIZE=$1

DATASET_PATH=$(get_real_path $2)
PRE_TRAINED_PATH=$(get_real_path $3)

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
    exit 1
fi

if [ ! -f $PRE_TRAINED_PATH ] && [ $# == 4 ]
then
    echo "error: PRE_TRAINED_PATH=$PRE_TRAINED_PATH is not a file"
    exit 1
fi

if [ -d "./distributed_train" ]
then
    rm -rf ./distributed_train
    echo "Remove dir ./distributed_train"
fi
mkdir ./distributed_train
echo "Create a dir ./distributed_train"
cp ./train.py ./distributed_train
cp -r ./src ./distributed_train
cd ./distributed_train || exit
echo "Start training for $DEVICE_NUM devices"
env > env.log

mpirun -n $RANK_SIZE --allow-run-as-root \
  --output-filename log_output \
  --merge-stderr-to-stdout \
  python train.py \
  --is_distributed True \
  --platform GPU \
  --data_url $DATASET_PATH \
  --train_url train_output \
  --device_num $DEVICE_NUM > log 2>&1 &
cd ..
