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

if [ $# != 3 ]; then
  echo "Usage: bash ./scripts/run_standalone_train.sh [DEVICE_ID] [TRAIN_DATA_DIR] [EVAL_DATA_DIR]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export DEVICE_ID=$1
export RANK_SIZE=1

TRAIN_DATA_DIR=$(get_real_path $2)
echo $TRAIN_DATA_DIR

if [ ! -d $TRAIN_DATA_DIR ]
then
    echo "error: TRAIN_DATA_DIR=$TRAIN_DATA_DIR is not a directory."
exit 1
fi

EVAL_DATA_DIR=$(get_real_path $3)
echo $EVAL_DATA_DIR

if [ ! -d $EVAL_DATA_DIR ]
then
    echo "error: EVAL_DATA_DIR=$EVAL_DATA_DIR is not a directory."
exit 1
fi

python ./train.py  \
    --train_dataset_path=$TRAIN_DATA_DIR \
    --eval_dataset_path=$EVAL_DATA_DIR \
    --run_distribute=False > log.txt 2>&1 &