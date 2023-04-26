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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash run_eval_gpu.sh [DATASET] [DEVICE_ID] [CHECKPOINT_PATH](optional)"
exit 1
fi

DATASET=$1
echo $DATASET

export DEVICE_NUM=1
export DEVICE_ID=$2
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# == 3 ]
then
    CHECKPOINT_PATH=$(get_real_path $3)
    if [ ! -f $CHECKPOINT_PATH ]
    then
        echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file"
    exit 1
    fi
fi

if [ -d "eval$2" ];
then
    rm -rf ./eval$2
fi

mkdir ./eval$2
cp ../*.py ./eval$2
cp ../*.yaml ./eval$2
cp -r ../src ./eval$2
cd ./eval$2 || exit

env > env.log
echo "start inferring for device $DEVICE_ID"

if [ $# == 2 ]
then
  python eval.py \
      --run_platform="GPU" \
      --dataset=$DATASET \
      --device_id=$2 > eval_log.txt 2>&1 &
fi

if [ $# == 3 ]
then
  python eval.py \
      --run_platform="GPU" \
      --dataset=$DATASET \
      --checkpoint_path=$CHECKPOINT_PATH \
      --device_id=$2 > eval_log.txt 2>&1 &
fi

cd ..