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

if [ $# != 5 ] && [ $# != 6 ]
then
    echo "Using: bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]"
    echo "or" 
    echo "Using: bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $2)    # dataset_path
PATH2=$(get_real_path $3)    # train_output_path 
PATH3=$(get_real_path $4)    # pretrained or resume ckpt_path

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory."
    exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: TRAIN_OUTPUT_PATH=$PATH2 is not a directory."
fi

if [ ! -f $PATH3 ]
then
    echo "error: CHECKPOINT_PATH=$PATH3 is not a file."
    exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1

if [ -d "./train" ]
then
    rm -rf ./train
    echo "Remove dir ./train."
fi
mkdir ./train
echo "Create a dir ./train."
cp ./train.py ./train
cp -r ./src ./train
cd ./train || exit
echo "Start training for device $DEVICE_ID"
env > env.log
if [ $# == 5 ]
then
    python3 train.py --data_path $PATH1 --output_path $PATH2 --checkpoint_path $PATH3 --modelarts False --run_distribute False --device_target Ascend --lr 0.0017 --lr_power 6e-10 --begin_epoch 0 --eval_callback $5 &> log &
else
    python3 train.py --data_path $PATH1 --output_path $PATH2 --checkpoint_path $PATH3 --modelarts False --run_distribute False --device_target Ascend --lr 0.0017 --lr_power 6e-10 --begin_epoch $5 --eval_callback $6 &> log &
fi
cd ..

