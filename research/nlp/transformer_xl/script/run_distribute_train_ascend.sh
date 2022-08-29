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

if [ $# != 6 ]; then
  echo "Usage: bash run_distribute_train_ascend.sh [DEVICE_NUM] [RANK_TABLE_FILE]
   [DATA_DIR] [DATA_NAME] [TRAIN_URL] [CONFIG_PATH]"
  exit 1
fi

if [ $1 -lt 1 ] || [ $1 -gt 8 ]; then
  echo "error: DEVICE_NUM=$1 is not in (1-8)"
  exit 1
fi

DATA_DIR=$3
DATA_NAME=$4
TRAIN_URL=$5
CONFIG_PATH=$6

echo "DATA_DIR="$DATA_DIR
echo "DATA_NAME="$DATA_NAME
echo "TRAIN_URL="$TRAIN_URL
echo "CONFIG_PATH="$CONFIG_PATH

export CONFIG_PATH=${CONFIG_PATH}
export DEVICE_NUM=$1
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2

BASEPATH=$(
  cd "$(dirname $0)" || exit
  pwd
)

export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "./train" ]; then
  rm -rf ./train
fi
mkdir ./train
cd ./train || exit

echo "Start Training :)"

for ((i=0;i<$RANK_SIZE;i++))
do
  export DEVICE_ID=$i
  export RANK_ID=$i
  export GLOV_v=1
  python -u ${BASEPATH}/../train.py --device_target="Ascend" --datadir=$DATA_DIR --dataset=$DATA_NAME --train_url=$TRAIN_URL > train_ascend_$i.log 2>&1 &
done
