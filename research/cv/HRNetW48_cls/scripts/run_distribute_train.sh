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

if [ $# -lt 3 ] || [ $# -gt 6 ]
then
    echo "Using: bash scripts/run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH]"
    echo "or"
    echo "Using: bash scripts/run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [EVAL_DATASET_PATH]"
    echo "or"
    echo "Using: bash scripts/run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH]"
    echo "or"
    echo "Using: bash scripts/run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_DATASET_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

RANK_PATH=$(get_real_path $1)                     # rank_file_path
DATA_PATH=$(get_real_path $2)                     # dataset_path
TRAIN_PATH=$(get_real_path $3)                    # train_output_path

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a directory."
    exit 1
fi

if [ ! -d $TRAIN_PATH ]
then
    mkdir $TRAIN_PATH
else
    rm -rf $TRAIN_PATH
    mkdir $TRAIN_PATH
fi

if [ $# == 4 ]
then
    EVAL_DATASET_PATH=$(get_real_path $4)     # eval_dataset_path

    if [ ! -d $EVAL_DATASET_PATH ]
    then
        echo "error: EVAL_DATASET_PATH=$EVAL_DATASET_PATH is not a directory."
        exit 1
    fi
fi

if [ $# == 5 ]
then
    CKPT_PATH=$(get_real_path $4)             # checkpoint_path
    BEGIN_EPOCH=$5                            # begin epoch

    if [ ! -f $CKPT_PATH ]
    then
        echo "error: CKPT_PATH=$CKPT_PATH is not a file."
        exit 1
    fi
fi

if [ $# == 6 ]
then
    CKPT_PATH=$(get_real_path $4)             # checkpoint_path
    BEGIN_EPOCH=$5                            # begin epoch
    EVAL_DATASET_PATH=$(get_real_path $6)     # eval_dataset_path

    if [ ! -d $EVAL_DATASET_PATH ]
    then
        echo "error: EVAL_DATASET_PATH=$EVAL_DATASET_PATH is not a directory."
        exit 1
    fi

    if [ ! -f $CKPT_PATH ]
    then
        echo "error: CKPT_PATH=$CKPT_PATH is not a file."
        exit 1
    fi
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_PATH

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
    if [ $# == 3 ]; then
        python train.py --data_path=$DATA_PATH --train_path=$TRAIN_PATH --run_distribute=True &> log &
    elif [ $# == 4 ]; then
        python train.py --data_path=$DATA_PATH --train_path=$TRAIN_PATH --eval_data_path=$EVAL_DATASET_PATH --run_distribute=True &> log &
    elif [ $# == 5 ]; then
        python train.py --data_path=$DATA_PATH --train_path=$TRAIN_PATH --checkpoint_path=$CKPT_PATH --begin_epoch=$BEGIN_EPOCH --run_distribute=True &> log &
    else
        python train.py --data_path=$DATA_PATH --train_path=$TRAIN_PATH --eval_data_path=$EVAL_DATASET_PATH --checkpoint_path=$CKPT_PATH --begin_epoch=$BEGIN_EPOCH --run_distribute=True &> log &
    fi

    cd ..
done
