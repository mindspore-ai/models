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
    echo "Using: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]"
    echo "or" 
    echo "Using: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $1)    # rank_table_file
PATH2=$(get_real_path $2)    # dataset_path
PATH3=$(get_real_path $3)    # train_output_path 
PATH4=$(get_real_path $4)    # pretrained or resume ckpt_path

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file."
    exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: DATASET_PATH=$PATH2 is not a directory."
    exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: TRAIN_OUTPUT_PATH=$PATH3 is not a directory."
fi

if [ ! -f $PATH4 ]
then
    echo "error: CHECKPOINT_PATH=$PATH4 is not a file."
    exit 1
fi

export DEVICE_NUM=4
export RANK_SIZE=4
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))


for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ./train.py ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "Start training for rank $RANK_ID, device $DEVICE_ID."
    env > env.log
    if [ $# == 5 ]
    then
        python3 train.py --data_path $PATH2 --output_path $PATH3 --checkpoint_path $PATH4 --modelarts False --run_distribute True --device_target Ascend --lr 0.0012 --lr_power 6e-10 --begin_epoch 0 --end_epoch 1000 --eval_callback $5 --eval_interval 50 &> log &
    elif [ $# == 6 ]
    then
        python3 train.py --data_path $PATH2 --output_path $PATH3 --checkpoint_path $PATH4 --modelarts False --run_distribute True --device_target Ascend --lr 0.0012 --lr_power 6e-10 --begin_epoch $5 --end_epoch 1000 --eval_callback $6 --eval_interval 50 &> log &
    fi
    cd ..
done
