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

if [ $# != 2 ] && [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH](optional) [NOT_MASK](optional)"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $1)
PATH2=$2

echo "path1: $PATH1"
echo "path2: $PATH2"

if [ $# == 3 ] || [ $# == 4 ]
then
    PATH3=$3
    echo "path3: $PATH3"
fi
not_mask=$4

if [ ! -d $PATH1 ]
then
    echo "error: DATA_PATH=$PATH1 is not a folder"
    exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8

cd ..
if [ ! -d "./log_dist" ]
then
    echo "pwd: $(pwd)"
    mkdir ./log_dist
fi
cd ./log_dist
echo "pwd: $(pwd)"

if [ $# == 3 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python -u ../train.py --device_target="GPU" --do_train=True --run_distribute=True --device_num=$DEVICE_NUM \
    --coco_root=$PATH1 --mindrecord_dir=$PATH2 --pre_trained=$PATH3 >> train_dist.log 2>&1 &
fi

if [ $# == 2 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python -u ../train.py --device_target="GPU" --do_train=True --run_distribute=True --device_num=$DEVICE_NUM \
    --coco_root=$PATH1 --mindrecord_dir=$PATH2 >> train_dist.log 2>&1 &
fi

if [ $# == 4 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python -u ../train.py --device_target="GPU" --do_train=True --run_distribute=True --device_num=$DEVICE_NUM \
    --not_mask=$not_mask --coco_root=$PATH1 --mindrecord_dir=$PATH2 >> train_dist.log 2>&1 &
fi
