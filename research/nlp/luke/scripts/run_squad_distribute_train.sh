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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_distributed_train_ascend.sh RANK_TABLE_ADDR PRE_TRAIN_DATASET"
echo "for example:"
echo "sh run_distributed_train_ascend.sh \
  /home/workspace/rank_table_8p.json \
  /home/workspace/squad_data \
  /home/workspace/pre_luke/luke_large "
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_TABLE_ADDR=$1
DATA=$2
MODEL_FILE=$3

current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_TABLE_FILE=$RANK_TABLE_ADDR
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_ADDR

echo $RANK_TABLE_FILE
echo $DATA
echo $MODEL_FILE

export RANK_SIZE=8
export GLOG_v=2

for((i=0;i<${RANK_SIZE};i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../*.py .
    cp -r ../src .
    export RANK_ID=$i
    export DEVICE_ID=$i
  python ./run_squad_train.py \
    --data $DATA \
    --model_file $MODEL_FILE \
    --warmup_proportion 0.09 \
    --num_train_epochs 2 \
    --train_batch_size 2 \
    --learning_rate 12e-6 \
    --duoka True \
    --dataset_sink_mode True &> train_squad${i}.log &
    cd ${current_exec_path} || exit
done
echo "start to train, please wait"