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

if [ $# != 1 ] && [ $# != 2 ]
then 
    echo "Usage: bash run_distribute_train.sh [DATA_PATH] [PRETRAINED_PATH](optional)"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

if [ ! -z "${2}" ];
then
  PATH2=$(get_real_path $2)
else
  PATH2=$2
fi

PATH1=$(get_real_path $1)

echo $PATH1

if [ $# == 2 ]
then
    echo $PATH2
fi

if [ ! -d $PATH1 ]
then 
    echo "error: DATA_PATH=$PATH1 is not a folder"
    exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8

echo 3 > /proc/sys/vm/drop_caches

if [ -d "train_parallel" ];
then
    rm -rf ./train_parallel
fi

mkdir ./train_parallel
cp ../*.py ./train_parallel
cp -r ../src ./train_parallel
cp ../*.yaml ./train_parallel
cp *.sh ./train_parallel
cd ./train_parallel || exit
env > env.log
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python train.py --device_target="GPU" --do_train=True --run_distribute=True --device_num=$DEVICE_NUM \
  --coco_root=$PATH1 --pre_trained=$PATH2 &> log &
cd ..
