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

if [ $# != 3 ]
then
    echo "Usage: bash run_eval.sh [EVAL_DIR] [DEVICE_ID] [PRETRAINED_BACKBONE]"
    exit 1
fi

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

SCRIPT_NAME='eval.py'

ulimit -c unlimited

EVAL_DIR=$(get_real_path $1)
DEVICE_ID=$2
PRETRAINED_BACKBONE=$(get_real_path $3)

if [ ! -f $PRETRAINED_BACKBONE ]
    then
    echo "error: PRETRAINED_PATH=$PRETRAINED_BACKBONE is not a file"
exit 1
fi

echo 'start evaluating'
export RANK_ID=0


echo 'start device '$DEVICE_ID

if [ ! -d "${current_exec_path}/device$DEVICE_ID" ]; then
  mkdir ${current_exec_path}/device$DEVICE_ID
fi

cd ${current_exec_path}/device$DEVICE_ID || exit

dev=`expr $DEVICE_ID + 0`
export DEVICE_ID=$dev
python3 ${current_exec_path}/${SCRIPT_NAME} \
    --eval_dir=$EVAL_DIR \
    --pretrained=$PRETRAINED_BACKBONE > eval.log  2>&1 &

echo 'running'
