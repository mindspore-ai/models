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

if [[ $# != 5 ]]; then
    echo "Usage: bash run_distribution_train_cifar10.sh [RANK_TABLE][RANK_SIZE][DEVICE_START][DATASET_DIR][RESULT_DIR]"
exit 1
fi

get_real_path(){
  if [ -z $1 ]; then
    echo "error: RANK_TABLE is empty"
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

ulimit -u unlimited
export RANK_SIZE=$2
RANK_TABLE_FILE=$(realpath $1)
export RANK_TABLE_FILE
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

device_start=$3
for((i=0; i<${RANK_SIZE}; i++))
do
    export DEVICE_ID=$((device_start + i))
    export RANK_ID=$i
    rm -rf ./device$i
    mkdir ./device$i
    cp -r ../src ./device$i
    cp ../*.py ./device$i
    echo "Start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./device$i
    env > env.log
    python train.py  --batch_size 32 --dataset cifar10 --n_epochs 200 --train_data_path $DATASET_DIR --test_data_path $DATASET_DIR --result_path $5 --device_id=$i --distributed 1 > train.log 2>&1 &
    cd ..
done
