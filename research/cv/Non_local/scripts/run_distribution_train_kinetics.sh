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
if [[ $# != 8 ]]; then
    echo "Usage: bash run_distribution_train_kinetics.sh [RANK_TABLE][RANK_SIZE][DEVICE_START][ROOT_PATH][TRAIN_DATA_PATH][TEST_DATA_PATH][ANNOTATION_PATH][RESULT_PATH]"
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

ROOT_PATH=$(get_real_path $4)

if [ ! -d $ROOT_PATH ]
then
    echo "error: ROOT_PATH=$ROOT_PATH is not a directory"
exit 1
fi

TRAIN_DATA_PATH=$(get_real_path $5)

if [ ! -d $TRAIN_DATA_PATH ]
then
    echo "error: TRAIN_DATA_PATH=$TRAIN_DATA_PATH is not a directory"
exit 1
fi

TEST_DATA_PATH=$(get_real_path $6)

if [ ! -d $TEST_DATA_PATH ]
then
    echo "error: TEST_DATA_PATH=$TEST_DATA_PATH is not a directory"
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
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cp ../train.py ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./train_parallel$i ||exit
    env > env.log\
    python train.py --root_path $ROOT_PATH \
                    --train_data_path $TRAIN_DATA_PATH \
                    --test_data_path $TEST_DATA_PATH \
                    --annotation_path $7 \
                    --result_path $8 \
                    --batch_size 16 \
                    --dataset kinetics \
                    --sample_size 224 \
                    --sample_duration 32 \
                    --n_epochs 100 \
                    --n_threads 8 \
                    --learning_rate 0.01 \
                    --distributed 1  > train.log 2>&1 &
    cd ..
done
