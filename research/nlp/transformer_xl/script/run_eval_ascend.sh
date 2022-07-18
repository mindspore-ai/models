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

if [ $# -lt 4 ] ||  [ $# -gt 5 ]
then
    echo "Usage: bash run_eval_ascend.sh [DATA_DIR] [DATA_NAME] [CKPT_PATH] [CONFIG_PATH] [DEVICE_ID(optional)]"
exit 1
fi

export DEVICE_ID=0

if [ $# = 5 ] ; then
  export DEVICE_ID=$5
fi;


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_DIR=$(get_real_path $1)

if [ ! -d $DATA_DIR ]
then
    echo "error: DATA_DIR=$DATA_DIR is not a directory"
exit 1
fi

DATA_NAME=$2
CKPT_PATH=$3
CONFIG_PATH=$4

echo "DATA_DIR="$DATA_DIR
echo "DATA_NAME="$DATA_NAME
echo "CKPT_PATH="$CKPT_PATH
echo "CONFIG_PATH="$CONFIG_PATH

export CONFIG_PATH=${CONFIG_PATH}
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0
if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval

env > env.log

echo "Start evaluation for device $DEVICE_ID :)"

python ../eval.py --device_id=$DEVICE_ID --datadir=$DATA_DIR --dataset=$DATA_NAME --ckpt_path=$CKPT_PATH --device_target="Ascend" &> eval.log &
