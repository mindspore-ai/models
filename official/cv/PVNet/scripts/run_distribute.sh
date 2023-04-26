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

if [ $# != 1 ]
then
    echo "Usage: bash scripts/run_distribute.sh [RANK_TABLE_FILE]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
if [ ! -f $PATH1 ]
then
  echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
  exit 1
fi

export RANK_SIZE=8

export RANK_TABLE_FILE=$PATH1
echo RANK_TABLE_FILE=${RANK_TABLE_FILE}

current_exec_path=$(pwd)
echo ${current_exec_path}

echo 'start training'
for((i=0;i<=$RANK_SIZE-1;i++));
do
    echo 'start rank '$i
    if [ -d ${current_exec_path}/device$i ]; then
      echo ${current_exec_path}/device$i
      rm -rf ${current_exec_path}/device$i
    fi

    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i
    export RANK_ID=$i
    dev=`expr $i`
    export DEVICE_ID=$dev
    python ../train.py \
        --epoch_size 200 --batch_size 32 --workers_num 12 --distribute 1 > train.log  2>&1 &
done