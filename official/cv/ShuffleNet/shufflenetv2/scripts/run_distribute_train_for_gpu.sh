#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
if [ $# -lt 3 ] || [ $# -gt 4 ]
then
    echo "Usage: 
          sh run_distribute_train_for_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [PRETRAINED_CKPT_FILE](optional)
          "
exit 1
fi

# check device number
if [ "$1" -eq "$1" ] 2>/dev/null
then
    if [ $1 -lt 1 ] || [ $1 -gt 8 ]
    then
        echo "error: DEVICE_NUM=$1 is not in (1-8)"
    exit 1
    fi
else
    echo "error: DEVICE_NUM=$1 is not a number"
exit 1
fi

# check dataset path
if [ ! -d $3 ]
then
    echo "error: DATASET_PATH=$3 is not a directory"    
exit 1
fi

# check PRETRAINED_CKPT_FILE
if [ $# == 4 ] && [ ! -f $4 ]
then
    echo "error: PRETRAINED_CKPT_FILE=$4 is not a file"    
exit 1
fi

export DEVICE_NUM=$1
export RANK_SIZE=$1

if [ -d "../train" ];
then
    rm -rf ../train
fi
mkdir ../train
cd ../train || exit

export CUDA_VISIBLE_DEVICES="$2"

if [ $# == 3 ]
then
    mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python ../train.py --platform='GPU' --is_distributed=True --enable_tobgr=True --normalize=False --use_nn_default_loss=False --dataset_path=$3 > train.log 2>&1 &
fi

if [ $# == 4 ]
then
    mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python ../train.py --platform='GPU' --is_distributed=True --enable_tobgr=True --normalize=False --use_nn_default_loss=False --dataset_path=$3 --resume=$4 > train.log 2>&1 &
fi

