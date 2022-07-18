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

if [ $# -ne 5 ]; then
    echo "Usage: bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_DIR] [DATA_NAME]
     [TRAIN_URL] [CONFIG_PATH]"
exit 1
fi

DEVICE_ID=$1
DATA_DIR=$2
DATA_NAME=$3
TRAIN_URL=$4
CONFIG_PATH=$5

echo "DATA_DIR="$DATA_DIR
echo "DATA_NAME="$DATA_NAME
echo "TRAIN_URL="$TRAIN_URL
echo "CONFIG_PATH="$CONFIG_PATH

export CONFIG_PATH=${CONFIG_PATH}

if [ -d "./train_stand" ]; then
  rm -rf ./train_stand
fi
mkdir ./train_stand
cd ./train_stand || exit

echo "Start training for device $DEVICE_ID :)"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python ../../train.py --device_target="GPU" --datadir=$DATA_DIR --dataset=$DATA_NAME --train_url=$TRAIN_URL > train_stand_gpu.log 2>&1 &
