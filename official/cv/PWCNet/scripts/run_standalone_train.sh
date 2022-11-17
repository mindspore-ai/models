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

echo "Usage: bash run_standalone_train.sh [TRAIN_LABEL_FILE] [EVAL_DIR] [DEVICE_ID] [PRETRAINED_BACKBONE]"



get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

current_exec_path=$(pwd)

dirname_path=$(dirname "$(pwd)")

export PYTHONPATH=${dirname_path}:$PYTHONPATH

export RANK_SIZE=1

SCRIPT_NAME='train.py'

ulimit -c unlimited

TRAIN_LABEL_FILE=$(get_real_path $1)
EVAL_DIR=$(get_real_path $2)
DEVICE_ID=$3
PRETRAINED_BACKBONE=''

if [ ! -d $TRAIN_LABEL_FILE ]
then
    echo "error: TRAIN_LABEL_FILE=$TRAIN_LABEL_FILE is not a file"
exit 1
fi

if [ ! -d $EVAL_DIR ]
then
    echo "error: EVAL_DIR=$EVAL_DIR is not a file"
exit 1
fi


PRETRAINED_BACKBONE=$(get_real_path $4)
if [ ! -f $PRETRAINED_BACKBONE ]
then
    echo "error: PRETRAINED_PATH=$PRETRAINED_BACKBONE is not a file"
exit 1
fi

echo 'start training'
export RANK_ID=0

echo 'start device '$DEVICE_ID

if [ ! -d "${current_exec_path}/device$DEVICE_ID" ]; then
  mkdir ${current_exec_path}/device$DEVICE_ID
fi


cd ${current_exec_path}/device$DEVICE_ID || exit

dev=`expr $DEVICE_ID + 0`

export DEVICE_ID=$dev
python3 ${current_exec_path}/${SCRIPT_NAME} \
    --is_distributed=0 \
    --batch_size=4 \
    --train_label_file=$TRAIN_LABEL_FILE \
    --eval_dir=$EVAL_DIR \
    --pretrained=$PRETRAINED_BACKBONE > train.log  2>&1 &

echo 'running'
