#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
if [ $# != 6 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_ascend.sh DEVICE_NUM EPOCH_SIZE DATA_PATH RANK_TABLE_FILE CONFIG_PATH CHECKPOINT_PATH "
echo "for example: bash run_distribute_train_ascend.sh 8 52 /path/ende-l128-mindrecord00 /path/hccl.json ./default_config_large.yaml /path/checkpoint"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

rm -rf run_distribute_train
mkdir run_distribute_train
cd run_distribute_train || exit

EPOCH_SIZE=$2
DATA_PATH=$3
CHECKPOINT_PATH=$6
start=0

export HCCL_CONNECT_TIMEOUT=600
export RANK_TABLE_FILE=$4
export CONFIG_PATH=$5
export RANK_SIZE=$1
export HCCL_FLAG=1
export DEPLOY_MODE=0

for((i=0;i<RANK_SIZE;i++))
do
    
    export DEVICE_ID=$((i+start))
    export RANK_ID=$i
    export GE_USE_STATIC_MEMORY=1

    mkdir helper$i
    cp -rf ../../src/ ../../train.py ../../*.yaml ./helper$i
    cd ./helper$i || exit
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log
    nohup python train.py  \
    --config_path=$CONFIG_PATH \
    --epoch=$EPOCH_SIZE \
    --batch_size=32 \
    --device_id=$DEVICE_ID \
    --checkpoint_path=$CHECKPOINT_PATH \
    --save_checkpoint_steps=50 \
    --keep_checkpoint_max=4 \
    --data_path=$DATA_PATH > log_distributed.txt 2>&1 &
    cd ../
done
cd ..