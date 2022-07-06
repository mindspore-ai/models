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

if [ $# != 4 ]
then
    echo "Usage: bash run_distribute_train.sh [CFG_PATH] [SAVE_PATH] [RANK_SIZE] [RANK_TABLE]"
exit 1
fi

CFG_PATH=$1
SAVE_PATH=$2
RANK_SIZE=$3
RANK_TABLE=$4

export RANK_TABLE_FILE=$RANK_TABLE
export RANK_SIZE=$3
echo "RANK_TABLE_FILE=$RANK_TABLE_FILE"
current_exec_path=$(pwd)
echo ${current_exec_path}

if [ -d $SAVE_PATH ];
then
    rm -rf $SAVE_PATH
fi
mkdir -p $SAVE_PATH

cp $CFG_PATH $SAVE_PATH

for((i=0;i<=$RANK_SIZE-1;i++));
do
    echo 'start rank '$i
    if [ -d ${current_exec_path}/device$i ];
    then
        rm -rf ${current_exec_path}/device$i
    fi
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    export RANK_ID=$i
    dev=`expr $i + 0`
    export DEVICE_ID=$dev
    python ${current_exec_path}/train.py \
      --is_distributed=1 \
      --device_target=Ascend \
      --cfg_path=$CFG_PATH \
      --save_path=$SAVE_PATH > train_log$i.txt 2>&1 &
done

