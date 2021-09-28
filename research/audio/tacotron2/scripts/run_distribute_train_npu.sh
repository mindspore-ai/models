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
DATA_DIR=$1
export RANK_TABLE_FILE=${ROOT_PATH}/$2
export MINDSPORE_HCCL_CONFIG_PATH=${ROOT_PATH}/$2
export HCCL_CONNECT_TIMEOUT=600
export RANK_SIZE=$3
begin=$4
log_dir=${ROOT_PATH}/logs
if [ ! -d $log_dir ]; then
    mkdir $log_dir
else
    echo $log_dir exist
fi
for((i=begin;i<begin + RANK_SIZE;i++))
do
    rm ${ROOT_PATH}/logs/device$i/ -rf
    mkdir ${ROOT_PATH}/logs/device$i
    cd ${ROOT_PATH}/logs/device$i || exit 
    export DEVICE_ID=$i
    let rank=$i-$begin
    export RANK_ID=$rank
    export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
    echo $i
    python3 ${ROOT_PATH}/train.py -d ${ROOT_PATH}/${DATA_DIR} --is_distributed 'true' -p ''>log$i.log 2>&1 &
done
