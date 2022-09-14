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

if [ $# != 3 ]; then
  echo "Usage: bash run_distribute_train_gpu.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR]"
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
echo $TRAIN_PATH

TEST_PATH=$(get_real_path $2)
echo $TEST_PATH

SAVE_DIR=$(get_real_path $3)
echo $SAVE_DIR

if [ ! -f $TRAIN_PATH ]; then
  echo "error: TRAIN_PATH=$TRAIN_PATH is not a file"
fi

if [ ! -f $TEST_PATH ]; then
  echo "error: TEST_PATH=$TEST_PATH is not a file"
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8

if [ -d "train_parallel" ]; then
  rm -rf ./train_parallel
fi

WORKDIR=./train_parallel
mkdir $WORKDIR
cp ../*.py $WORKDIR
cp -r ../src $WORKDIR
cp ../*.yaml $WORKDIR
cd $WORKDIR || exit

echo "Start training on $DEVICE_NUM GPU devices"
env > env.log

mpirun --allow-run-as-root -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout \
        python train.py \
        --train_path=$TRAIN_PATH \
        --test_path=$TEST_PATH \
        --save_dir=$SAVE_DIR \
        --device_target=GPU \
        --run_distribute=True > train.log 2>&1 &
cd ..
