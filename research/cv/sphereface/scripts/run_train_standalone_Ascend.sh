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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_train_Ascend.sh [DEVICE_ID] [PRE_TRAINED](optional)"
echo "=============================================================================================================="

if [ $# -lt 1 ]
then
    echo "Usage: bash run_train_cpu.sh [DEVICE_ID] [PRE_TRAINED](optional)"
    exit 1
fi
export DEVICE_ID=$1
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "./train_Ascend" ];
then
    rm -rf ./train_Ascend
fi
mkdir ./train_Ascend
cd ./train_Ascend || exit
mkdir src
cd ../
cp ../*.py ./train_Ascend
cp ../*.yaml ./train_Ascend
cp -r ../src ./train_Ascend/
cd ./train_Ascend


if [ -f $2 ]  # pretrained ckpt
then
        nohup python ${BASEPATH}/../train.py \
                --is_distributed=0 \
                --device_target='Ascend' \
                --device_id=$1 \
                --T_max=19 \
                --max_epoch=20 \
                --per_batch_size=256 \
                --train_pretrained=$2 > train.log 2>&1 &
else
        nohup python ${BASEPATH}/../train.py \
                --is_distributed=0 \
                --device_id=$1 \
                --T_max=19 \
                --max_epoch=20 \
                --per_batch_size=256 \
                --device_target='Ascend' > train.log 2>&1 &
fi
