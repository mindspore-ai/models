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
if [ $# != 2 ]
then
    echo "usage sh run_distribute_gpu.sh [device_id] [dataset_path]"
exit 1
fi

# check dataset file
if [ ! -d $2 ]
then
    echo "error: DATASET_PATH=$2 is not a directory"
exit 1
fi

#create train directory to save train.log
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
echo $BASEPATH
if [ -d "../train" ];
then
    rm -rf ../train
fi
mkdir ../train
cd ../train || exit

export CUDA_VISIBLE_DEVICES=$1
python ${BASEPATH}/../train.py \
        --dataset_path $2 \
        --device_target GPU > train.log 2>&1 &