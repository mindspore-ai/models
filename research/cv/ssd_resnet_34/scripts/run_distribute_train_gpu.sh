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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional) [PRE_TRAINED_EPOCHS](required if PRE_TRAINED_PATH is specified)"
echo "for example: bash scripts/run_distribute_train_gpu.sh 8 coco /home/ssd-coco /home/coco-mindrecord ./train_out /home/ssd-500_458.ckpt(optional) 500(required if PRE_TRAINED_PATH is specified)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 5 ] && [ $# != 7 ]
then
    echo "Using: bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH]"
    echo "or"
    echo "Using: bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH] [PRE_TRAINED_EPOCHS]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $3)    # dataset_path
PATH2=$(get_real_path $4)    # mindrecord_path
PATH3=$(get_real_path $5)    # train_output_path
PATH4=$(get_real_path $6)    # pre_trained_path


if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory."
    exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: MINDRECORD_PATH=$PATH2 is not a directory."
    exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: TRAIN_OUTPUT_PATH=$PATH3 is not a directory."
fi

if [ ! -d "$PATH3" ]
then
    mkdir "$PATH3"
    echo "Create a dir $PATH3."
fi

if [ ! -f $PATH4 ] && [ $# == 7 ]
then
    echo "error: PRE_TRAINED_PATH=$PATH4 is not a file."
    exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=$1
env > "$PATH3/env.log"

if [ $# == 5 ]
then
  mpirun -allow-run-as-root -n $DEVICE_NUM python train.py --data_url $PATH1 --mindrecord_url $PATH2 --train_url $PATH3 \
   --run_platform GPU --distribute True --lr 0.6 --mode sink --epoch_size 500 --batch_size 32 --dataset $2 \
   --save_checkpoint_epochs 1 > "$PATH3/train.log" 2>&1 &
fi

if [ $# == 7 ]
then
  mpirun -allow-run-as-root -n $DEVICE_NUM python train.py --data_url $PATH1 --mindrecord_url $PATH2 --train_url $PATH3 \
  --run_platform GPU --distribute True --lr 0.6 --mode sink --epoch_size 500 --batch_size 32 --dataset $2 \
  --save_checkpoint_epochs 1 --pre_trained $PATH4 --pre_trained_epochs $7 > "$PATH3/train.log" 2>&1 &
fi
cd ..
