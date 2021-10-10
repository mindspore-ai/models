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
if [ $# -lt 1 ] || [ $# -gt 2 ]
then
    echo "Usage: 
          sh run_standalone_train_for_gpu.sh [DATASET_PATH] [PRETRAINED_CKPT_FILE](optional) 
          "
exit 1
fi

# check dataset path
if [ ! -d $1 ]
then
    echo "error: DATASET_PATH=$1 is not a directory"    
exit 1
fi

# check PRETRAINED_CKPT_FILE
if [ $# == 2 ] && [ ! -f $2 ]
then
    echo "error: PRETRAINED_CKPT_FILE=$2 is not a file"    
exit 1
fi

if [ -d "../train" ];
then
    rm -rf ../train
fi
mkdir ../train
cd ../train || exit

if [ $# == 1 ]
then
    python ../train.py --platform='GPU' --enable_tobgr=True --normalize=False --use_nn_default_loss=False --dataset_path=$1 > train.log 2>&1 &
fi

if [ $# == 2 ]
then
    python ../train.py --platform='GPU' --enable_tobgr=True --normalize=False --use_nn_default_loss=False --dataset_path=$1 --resume=$2 > train.log 2>&1 &
fi
