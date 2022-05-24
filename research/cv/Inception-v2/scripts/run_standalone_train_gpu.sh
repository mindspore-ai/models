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
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_PATH]"
    echo "or bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_PATH] [PRE_TRAINED_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

DATASET_PATH=$(get_real_path $2)
PRE_TRAINED_PATH=$(get_real_path $3)

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
    exit 1
fi

if [ ! -f $PRE_TRAINED_PATH ] && [ $# == 3 ]
then
    echo "error: PRE_TRAINED_PATH=$PRE_TRAINED_PATH is not a file"
    exit 1
fi

export DEVICE_ID=$1
export DEVICE_NUM=1
export RANK_SIZE=1
export RANK_ID=0

if [ -d "./standalone_train" ]
then
    rm -rf ./standalone_train
    echo "Remove dir ./standalone_train"
fi
mkdir ./standalone_train
echo "Create a dir ./standalone_train."
cp ./train.py ./standalone_train
cp -r ./src ./standalone_train
cd ./standalone_train || exit
echo "Start training for device $DEVICE_ID"
env > env.log

if [ $# == 2 ]
then
    python train.py \
      --data_url $DATASET_PATH \
      --train_url train_output \
      --platform GPU \
      --device_num 1 > log 2>&1 &
else
    python train.py \
      --data_url $DATASET_PATH \
      --train_url train_output \
      --platform GPU --device_num 1 \
      --pre_trained $PRE_TRAINED_PATH > log 2>&1 &
fi
cd ..
