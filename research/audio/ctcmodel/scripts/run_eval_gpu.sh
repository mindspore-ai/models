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
  echo "Usage: bash run_eval_gpu.sh [TEST_PATH] [CHECKPOINT_PATH] [DEVICE_ID]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

TEST_PATH=$(get_real_path $1)
CHECKPOINT_PATH=$(get_real_path $2)
DEVICE_ID=$3

echo $TEST_PATH
echo $CHECKPOINT_PATH

if [ ! -f $TEST_PATH ]; then
  echo "error: TEST_PATH=$TEST_PATH is not a file"
  exit 1
fi

if [ ! -f $CHECKPOINT_PATH ]; then
  echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file"
  exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$DEVICE_ID
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval" ]; then
  rm -rf ./eval
fi

mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cd ./eval || exit

echo "start inferring for GPU $DEVICE_ID device"
env > env.log

python eval.py \
      --test_path=$TEST_PATH \
      --checkpoint_path=$CHECKPOINT_PATH \
      --device_id=$DEVICE_ID \
      --device_target=GPU \
      --beam=False > eval.log 2>&1 &
cd ..
