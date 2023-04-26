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
echo "bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional)"
echo "for example: bash scripts/run_distribute_train.sh /home/neu/hrnet_final/rank_table_file_path.json coco /home/neu/ssd-coco /home/neu/coco-mindrecord .train_out /home/neu/ssdresnet34lj/resnet34.ckpt(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 5 ] && [ $# != 6 ]
then
    echo "Using: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH]"
    echo "or"
    echo "Using: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH]"
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
PATH5=$(get_real_path $1)    # rank_table_file_path


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

if [ ! -f $PATH4 ] && [ $# == 6 ]
then
    echo "error: PRE_TRAINED_PATH=$PATH4 is not a file."
    exit 1
fi

if [ ! -f $PATH5 ]
then
    echo "error: RANK_TABLE_FILE_PATH=$PATH5 is not a file."
    exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH5

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))


for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    echo ./train_parallel$i
    cp ./train.py ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "Start training for rank $RANK_ID, device $DEVICE_ID."
    env > env.log
    if [ $# == 5 ]
    then
    python train.py --data_url $PATH1 --mindrecord_url $PATH2 --train_url $PATH3 --run_platform Ascend --lr 0.075 --epoch_size 500 --dataset $2 --distribute True --device_num 8 &> log &
    else
    python train.py --data_url $PATH1 --mindrecord_url $PATH2 --train_url $PATH3 --run_platform Ascend --lr 0.075 --epoch_size 500 --dataset $2 --pre_trained $PATH4 --distribute True --device_num 8 &> log &
    fi
    cd ..
done