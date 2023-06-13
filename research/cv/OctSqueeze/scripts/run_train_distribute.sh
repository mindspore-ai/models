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

if [ $# != 3 ]
then 
    echo "Usage: bash run_train_distribute.sh [TRAINING_DATASET_PATH] [CHECKPOINT_SAVE_PATH] [RANK_TABLE_FILE]"
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
if [ ! -d $PATH1 ]
then 
    echo "error: TRAINING_DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

data_path=$1
checkpoint=$2

export RANK_TABLE_FILE=$3
export RANK_SIZE=8

current_exec_path=$(pwd)
echo ${current_exec_path}

echo 'start training'
for((i=0;i<=$RANK_SIZE-1;i++));
do
    echo 'start rank '$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i
    export RANK_ID=$i
    dev=`expr $i`
    export DEVICE_ID=$dev
    python ../../train.py \
        --train=$data_path \
        --device_target='Ascend' \
        --checkpoint=$checkpoint \
        --is_distributed=1 > train.log  2>&1 &
done
