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
ROOT_PATH=$(pwd)
train_file=$1
DATA_DIR=$2
PRESET=$3
CKPT_DIR=$4
export RANK_TABLE_FILE=$5
export HCCL_CONNECT_TIMEOUT=600
export RANK_SIZE=$6
begin=$7
for((i=begin;i<begin+RANK_SIZE;i++))
do
    cd ${ROOT_PATH}/logs/device$i || exit
    let rank=$i-$begin
    export RANK_ID=$rank
    export DEVICE_ID=$i
    echo $i
    python3 ${ROOT_PATH}/${train_file} --data_path $DATA_DIR --preset $PRESET \
    --platform=Ascend --is_distributed --checkpoint_dir $CKPT_DIR >log$i.log 2>&1 &
done

