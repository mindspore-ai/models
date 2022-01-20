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

if [ $# != 5 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh   [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH]"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"


export LRPATH=$1
export GTPATH=$2
export VGGCKPT=$3
export VLRPATH=$4
export VGTPATH=$5


rm -rf ./train_standalone
mkdir ./train_standalone
cp -r ../src ./train_standalone
cp -r ../*.py ./train_standalone
cd ./train_standalone || exit

echo "start training"
env > env.log
if [ $# == 5 ]
then
        python train.py \
        --train_LR_path=$LRPATH --train_GT_path=$GTPATH --vgg_ckpt=$VGGCKPT \
        --val_LR_path=$VLRPATH --val_GT_path=$VGTPATH --platform=GPU > log 2>&1 &
fi
cd ..
