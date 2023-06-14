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

if [ $# != 4 ] && [ $# != 5 ]
then
    echo "Usage: bash run_train.sh [RANK_TABLE_FILE] [YAML_PATH] [DATA_URL] [TRAIN_URL] [INIT_WEIGHT][OPTIONAL]"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi
PATH1=$(realpath $1)
YAML_PATH=$(realpath $2)
DATA_URL=$(realpath $3)
TRAIN_URL=$(realpath $4)

if [ ! -d $TRAIN_URL ]; then
  mkdir $TRAIN_URL
fi
chmod -R 777 $TRAIN_URL

INIT_WEIGHT="${current_exec_path}/../pre_train_model/init_weight.ckpt"
if [ $# == 5 ]; then
    INIT_WEIGHT=$(realpath $5)
fi

ulimit -u unlimited
export RANK_TABLE_FILE=$PATH1
export RANK_SIZE=8
export HCCL_CONNECT_TIMEOUT=7200

current_exec_path=$(pwd)
echo ${current_exec_path}

echo 'start training'
for ((i = 0; i <= $RANK_SIZE - 1; i++)); do
  echo 'start rank '$i
  rm -rf ${current_exec_path}/device$i

  if [ ! -d ${current_exec_path}/device$i ]; then
    mkdir ${current_exec_path}/device$i
  fi

  cd ${current_exec_path}/device$i
  export RANK_ID=$i
  dev=$(expr $i)
  export DEVICE_ID=$dev
  python3 ../../train.py --yaml_path=$YAML_PATH --init_weight=$INIT_WEIGHT --data_url=$DATA_URL --train_url=$TRAIN_URL >train.log 2>&1 &
done
