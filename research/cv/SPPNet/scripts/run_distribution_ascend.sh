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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: bash run_distribution_ascend.sh [RANK_TABLE_FILE] [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [TRAIN_MODEL]"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$(get_real_path $2)
PATH2=$(get_real_path $3)


if [ ! -d $PATH1 ]
then
    echo "error: TRAIN_DATA_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: EVAL_DATA_PATH=$PATH2 is not a directory"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
RANK_TABLE_FILE=$(realpath $1)
export RANK_TABLE_FILE
export TRAIN_PATH=$2
export EVAL_PATH=$3
export TRAIN_MODEL=$4
export BASE_PATH=${TRAIN_MODEL}"_train_parallel"
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./$BASE_PATH$i
    mkdir ./$BASE_PATH$i
    cp -r ../src ./$BASE_PATH$i
    cp  ../train.py ./$BASE_PATH$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./$BASE_PATH$i ||exit
    env > env.log
    python train.py --is_distributed=1 --device_id=$i --train_path=$TRAIN_PATH --eval_path=$EVAL_PATH --device_num=$DEVICE_NUM --train_model=$TRAIN_MODEL > log 2>&1 &
    cd ..
done