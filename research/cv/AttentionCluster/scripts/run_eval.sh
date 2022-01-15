#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

if [ $# != 4 ]; then
    echo "Usage:
          bash run_eval.sh [DEVICE] [CONFIG] [CHECKPOINT_PATH] [DATASET_DIR]
         "
exit 1
fi

get_real_path(){
  if [ -z $1 ]; then
    echo "error: DATASET_DIR is empty"
    exit 1
  elif [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

DEVICE=$1
CONFIG=$(get_real_path $2)
CHECKPOINT_PATH=$(get_real_path $3)
DATASET_DIR=$(get_real_path $4)

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -f $CHECKPOINT_PATH ]
then
    echo "error: config=$CHECKPOINT_PATH is not a file."
exit 1
fi

if [ ! -d $DATASET_DIR ]
then
    echo "error: DATASET_PATH=$DATASET_DIR is not a directory"
exit 1
fi

echo "CONFIG: $CONFIG"
echo "CHECKPOINT: $CHECKPOINT_PATH"
echo "DATASET_DIR: $DATASET_DIR"
echo

if [ -d "$BASE_PATH/../eval" ];
then
    rm -rf $BASE_PATH/../eval
fi
mkdir $BASE_PATH/../eval
cd $BASE_PATH/../eval || exit

python $BASE_PATH/../eval.py  --device $DEVICE --config_path $CONFIG --ckpt $CHECKPOINT_PATH --data_dir $DATASET_DIR --result_dir . &> eval.log &
