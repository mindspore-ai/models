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
  echo "Usage: bash ./scripts/run_eval.sh [DEVICE_ID] [PATH_CHECKPOINT] [EVAL_DATA_DIR]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DEVICE_ID=$1

CHECKPOINT_PATH=$(get_real_path $2)
echo $CHECKPOINT_PATH

if [ ! -f $CHECKPOINT_PATH ]
then
    echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file."
exit 1
fi

DATA_DIR=$(get_real_path $3)
echo $DATA_DIR

if [ ! -d $DATA_DIR ]
then
    echo "error: DATA_DIR=$DATA_DIR is not a directory."
exit 1
fi

export RANK_SIZE=1

rm -rf evaluation_ascend
mkdir ./evaluation_ascend
cd ./evaluation_ascend || exit
echo  "start evaluating for device id $DEVICE_ID"
env > env.log
python ../eval.py  --dataset_path=$DATA_DIR --checkpoint_path=$CHECKPOINT_PATH --device_id=$DEVICE_ID > eval.log 2>&1 &
cd ../
