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
echo "bash run_distribute_train.sh DATA_PATH TRAIN_PATH BEGIN_DEVICE RANK_SIZE RANK_TABLE_FILE PRE_TRAINED(option)"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
TRAIN_PATH=$2
export TRAIN_PATH=${TRAIN_PATH}
RANK_TABLE_FILE=$5
BEGIN_DEVICE=$3
export RANK_SIZE=$4
export RANK_TABLE_FILE=$RANK_TABLE_FILE
PRE_TRAINED=$6

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HCCL_CONNECT_TIMEOUT=6000

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd ../
    cp ../train.py ./device$i
    cp ../*.yaml ./device$i
    cp -r ../src/ ./device$i
    cd ./device$i
    export DEVICE_ID=`expr $i + $BEGIN_DEVICE`
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    if [ $# -eq 5 ]
    then
        python train.py --data_path "$DATA_PATH" --output_path "$TRAIN_PATH"  --run_distribute True > train$i.log 2>&1 &
    fi
    if [ $# -eq 6 ]
    then
        python train.py --data_path "$DATA_PATH" --output_path "$TRAIN_PATH"  --run_distribute True --pre_trained_path $PRE_TRAINED > train$i.log 2>&1 &
    fi   
    echo "$i finish"
    cd ../
done
rm -rf device0
mkdir device0
cd ./device0
cd ../
cp ../train.py ./device0
cp ../*.yaml ./device0
cp -r ../src/ ./device0
cd ./device0
export DEVICE_ID=$BEGIN_DEVICE
export RANK_ID=0
echo "start training for device 0"
env > env0.log
if [ $# -eq 5 ]
then
    python train.py --data_path "$DATA_PATH" --output_path "$TRAIN_PATH"  --run_distribute True > train0.log 2>&1 &
fi
if [ $# -eq 6 ]
then
    python train.py --data_path "$DATA_PATH" --output_path "$TRAIN_PATH"  --run_distribute True --pre_trained_path $PRE_TRAINED > train0.log 2>&1 &
fi   
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../