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

if [ $# != 4 ]; then
  echo "Usage: bash run_distribute_train.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [RANK_TABLE_FILE]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
TRAIN_PATH=$(get_real_path $1)
TEST_PATH=$(get_real_path $2)
SAVE_DIR=$(get_real_path $3)
RANK_TABLE_FILE=$(get_real_path $4)
echo $TRAIN_PATH
echo $TEST_PATH
echo $SAVE_DIR
echo $RANK_TABLE_FILE

if [ ! -f $TRAIN_PATH ]; then
  echo "error: TRAIN_PATH=$TRAIN_PATH is not a file"
  exit 1
fi

if [ ! -f $TEST_PATH ]; then
  echo "error: TEST_PATH=$TEST_PATH is not a file"
  exit 1
fi

if [ ! -d $SAVE_DIR ]; then
  echo "error: SAVE_DIR=$SAVE_DIR is not a directory"
  exit 1
fi

if [ ! -f $RANK_TABLE_FILE ]; then
  echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
  exit 1
fi
ulimit -u unlimited
export HCCL_CONNECT_TIMEOUT=6000
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_FILE

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp ./*.py ./train_parallel$i
  cp ./*.yaml ./train_parallel$i
  cp -r ./src ./train_parallel$i
  cd ./train_parallel$i || exit
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  env >env.log
  python train.py \
    --train_path=$TRAIN_PATH \
    --test_path=$TEST_PATH \
    --save_dir=$SAVE_DIR \
    --device_target=Ascend \
    --run_distribute=True >train.log 2>&1 &
  cd ..
done
