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
if [ $# != 4 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh [CUB|AWA] [att|word] [DATA_PATH] [SAVE_CKPT]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$1  # one of [CUB|AWA]
TRAIN_MODE=$2  # one of [att|word]
DATA_PATH=$(get_real_path $3)  # dataset path
SAVE_CKPT=$(get_real_path $4)  # ../output

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a directory"
exit 1
fi

if [ ! -d $SAVE_CKPT  ];then
mkdir $SAVE_CKPT
fi

python train.py --dataset=$DATASET \
    --train_mode=$TRAIN_MODE \
    --data_path=$DATA_PATH \
    --save_ckpt=$SAVE_CKPT \
    --device_target=GPU > log_train_1P.txt 2>&1 &

