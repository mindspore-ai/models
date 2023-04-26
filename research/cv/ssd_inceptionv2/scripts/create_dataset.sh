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
echo "bash scripts/create_dataset.sh [DATASET_COCO2017] [DATASET_TARGET] [NUMBER_VAL_PATH]"
echo "It is better to use absolute path."
echo "================================================================================================================="
if [ $# != 3 ]
then
    echo "Using: bash scripts/create_dataset.sh [DATASET_COCO2017] [DATASET_TARGET] [NUMBER_VAL_PATH]"
    exit 1
fi
get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $1)    # coco2017 dataset_path
PATH2=$(get_real_path $2)    # path to store coco14mini
PATH3=$(get_real_path $3)    # coco14mini number_id_path

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_COCO2017=$PATH1 is not a directory."
    exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: DATASET_TARGET=$PATH2 is not a directory."
    exit 1
fi
if [ ! -d $PATH3 ]
then
    echo "error: DATASET_TARGET=$PATH3 is not a directory."
    exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1

python  train.py --data_complete False --data_url_raw $PATH1 --data_url $PATH2 --number_path $PATH3 &> log &
