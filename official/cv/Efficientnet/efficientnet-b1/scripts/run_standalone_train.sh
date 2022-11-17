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

if [ $# -lt 2 ] || [ $# -gt 5 ]
then
    echo "Using: bash scripts/run_standalone_train.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH]"
    echo "or"
    echo "Using: bash scripts/run_standalone_train.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH] [EVAL_DATASET_PATH]"
    echo "or"
    echo "Using: bash scripts/run_standalone_train.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH]"
    echo "or"
    echo "Using: bash scripts/run_standalone_train.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_DATASET_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

DATA_PATH=$(get_real_path $1)                     # dataset_path
TRAIN_PATH=$(get_real_path $2)                     # train_output_path

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

if [ $# == 3 ]
then
    EVAL_DATASET_PATH=$(get_real_path $3)     # eval_dataset_path

    if [ ! -d $EVAL_DATASET_PATH ]
    then
        echo "error: EVAL_DATASET_PATH=$EVAL_DATASET_PATH is not a directory."
        exit 1
    fi
fi

if [ $# == 4 ]
then
    CKPT_PATH=$(get_real_path $3)             # checkpoint_path
    BEGIN_EPOCH=$4                            # begin epoch

    if [ ! -f $CKPT_PATH ]
    then
        echo "error: CKPT_PATH=$CKPT_PATH is not a file."
        exit 1
    fi
fi

if [ $# == 5 ]
then
    CKPT_PATH=$(get_real_path $3)             # checkpoint_path
    BEGIN_EPOCH=$4                            # begin epoch
    EVAL_DATASET_PATH=$(get_real_path $5)     # eval_dataset_path

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

if [ $# == 2 ]; then
    python train.py --data_path=$DATA_PATH --train_path=$TRAIN_PATH &> log &
elif [ $# == 3 ]; then
    python train.py --data_path=$DATA_PATH --train_path=$TRAIN_PATH --eval_data_path=$EVAL_DATASET_PATH &> log &
elif [ $# == 4 ]; then
    python train.py --data_path=$DATA_PATH --train_path=$TRAIN_PATH --checkpoint_path=$CKPT_PATH --begin_epoch=$BEGIN_EPOCH &> log &
else
    python train.py --data_path=$DATA_PATH --train_path=$TRAIN_PATH --eval_data_path=$EVAL_DATASET_PATH --checkpoint_path=$CKPT_PATH --begin_epoch=$BEGIN_EPOCH &> log &
fi
