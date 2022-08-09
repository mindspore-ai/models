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

if [ $# -lt 3 ]; then
    echo "Usage: bash scripts/run_distributed_train_ascend.sh 
    [RANK_SIZE] [BEGIN] [RANK_TABLE_FILE]"
exit 1
fi

log_dir="./log"

if [ ! -d $log_dir ]; then
    mkdir log
fi

RANK_SIZE=$1  # num of cards
BEGIN=$2      # begin of the device, default 0
export RANK_TABLE_FILE=$3
export RANK_SIZE=$RANK_SIZE

pre_path=

for((i=$BEGIN;i<RANK_SIZE+BEGIN;i++))
do
    let rank=$i-$BEGIN
    export RANK_ID=$rank
    export DEVICE_ID=$i
    echo "start training for rank $rank, device $DEVICE_ID"
    if [ ! $pre_path ]; then
        python train.py --is_distributed --device_target 'Ascend' > log/distributed-train.log.$i 2>&1 &
    else
        python train.py --pre_trained_model_path $pre_path --is_distributed --device_target 'Ascend' > log/distributed-train.log.$i 2>&1 &
    fi  
done
