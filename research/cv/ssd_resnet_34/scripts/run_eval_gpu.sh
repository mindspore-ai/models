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
echo "bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [CHECKPOINT_PATH] [MINDRECORD_PATH]"
echo "for example: bash scripts/run_eval_gpu.sh 0 coco /home/ssd-coco /home/ssd_resnet34-990_458.ckpt /home/coco-mindrecord"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 5 ]
then
    echo "Using: bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [CHECKPOINT_PATH] [MINDRECORD_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $3)
PATH2=$(get_real_path $4)
PATH3=$(get_real_path $5)

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a dictionary."
    exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file."
    exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: MINDRECORD_PATH=$PATH3 is not a dictionary."
    exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$1
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi

echo "start evaluation for device $DEVICE_ID"
python eval.py --data_url $PATH1 --dataset $2 --device_id $1 --run_platform GPU \
--checkpoint_path $PATH2 --mindrecord $PATH3 &> eval.log &
cd ..