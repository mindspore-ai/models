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
echo "bash run_distribute_train.sh [DEVIDE_NUM] [RANK_TABLE_FILE] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [PRETRAIN_PATH]"
echo "For example:
bash run_distribute_train.sh 8 \\
~/rank_table_8pcs.json ucf101 \\
~/dataset/ucf101/jpg/ \\
~/dataset/ucf101/json/ucf101_01.json \\
~/results/ \\
~/pretrain_ckpt/pretrain.ckpt"
echo "It is better to use the ABSOLUTE path."
echo "=============================================================================================================="
set -e

if [ $# != 7 ]
then
  echo "Usage: bash run_distribute_train.sh [DEVIDE_NUM] [RANK_TABLE_FILE] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [PRETRAIN_PATH]"
exit 1
fi

DEVIDE_NUM=$1
RANK_TABLE_FILE=$2
DATASET=$3
VIDEO_PATH=$4
ANNOTATION_PATH=$5
RESULT_PATH=$6
PRETRAIN_PATH=$7

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
echo "RANK_TABLE_FILE: $RANK_TABLE_FILE"

export DEVIDE_NUM=$DEVIDE_NUM
export RANK_SIZE=$DEVIDE_NUM
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd ../
    cp ../*.py ./device$i
    cp ../*.yaml ./
    cp ../src/*.py ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py --is_modelarts False --run_distribute True --config_path \
    ../${DATASET}_config.yaml --video_path $VIDEO_PATH --annotation_path $ANNOTATION_PATH \
    --result_path $RESULT_PATH --pretrain_path $PRETRAIN_PATH > train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done
