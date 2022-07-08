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

if [ $# != 4 ] && [ $# != 5 ]
then
    echo "Using: bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH]"
    echo "or"
    echo "Using: bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH2=$(get_real_path $2)    # dataset_path
PATH3=$(get_real_path $3)    # train_output_path 
PATH4=$(get_real_path $4)    # pretrained or resume ckpt_path

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

export RANK_SIZE=$1
echo $RANK_SIZE

if [ $# == 4 ]
then
  export BEGIN_EPOCH=0
else
  export BEGIN_EPOCH=$5
fi

mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
  python3 train.py \
    --data_path $PATH2 \
    --output_path $PATH3 \
    --checkpoint_path $PATH4 \
    --modelarts False \
    --device_target GPU \
    --run_distribute True \
    --lr 0.0006 \
    --lr_power 6e-10 \
    --begin_epoch $BEGIN_EPOCH \
    --end_epoch 1000 \
    --eval_callback False \
    --eval_interval 50 &> out.train.log &
