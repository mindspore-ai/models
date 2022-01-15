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

if [[ $# -gt 6 ]]; then
    echo "Usage: bash run_standalone_train.sh [FC][NATT][EPOCHS][DATASET_DIR][RESULT_DIR][DEVICE_ID]"
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
DATASET_DIR=$(get_real_path $4)

if [ ! -d $DATASET_DIR ]
then
    echo "error: DATASET_PATH=$DATASET_DIR is not a directory"
exit 1
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

if [ -d "$BASE_PATH/../train" ];
then
    rm -rf $BASE_PATH/../train
fi
mkdir $BASE_PATH/../train
cd $BASE_PATH/../train || exit

python $BASE_PATH/../train.py --fc $1 --natt $2 --epochs $3 --data_dir $DATASET_DIR --result_dir $5 --device_id $6 --device Ascend &> train.log &