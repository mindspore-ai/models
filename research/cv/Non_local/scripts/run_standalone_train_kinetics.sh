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
if [[ $# != 6 ]]; then
    echo "Usage: bash run_standalone_train_kinetics.sh [ROOT_PATH][TRAIN_DATA_PATH][TEST_DATA_PATH][ANNOTATION_PATH][RESULT_PATH][DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ -z $1 ]; then
    echo "error: TRAIN_DATA_PATH is empty"
    exit 1
  elif [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ROOT_PATH=$(get_real_path $1)

if [ ! -d $ROOT_PATH ]
then
    echo "error: ROOT_PATH=$ROOT_PATH is not a directory"
exit 1
fi

TRAIN_DATA_PATH=$(get_real_path $2)

if [ ! -d $TRAIN_DATA_PATH ]
then
    echo "error: TRAIN_DATA_PATH=$TRAIN_DATA_PATH is not a directory"
exit 1
fi

TEST_DATA_PATH=$(get_real_path $3)

if [ ! -d $TEST_DATA_PATH ]
then
    echo "error: TEST_DATA_PATH=$TEST_DATA_PATH is not a directory"
exit 1
fi

python train.py --root_path $ROOT_PATH \
                --train_data_path $TRAIN_DATA_PATH \
                --test_data_path $TEST_DATA_PATH \
                --annotation_path $4 \
                --result_path $5 \
                --batch_size 16 \
                --dataset kinetics \
                --sample_size 224 \
                --sample_duration 32 \
                --n_epochs 100 \
                --n_threads 8 \
                --learning_rate 0.01 \
                --distributed 0 \
                --device_id $6
