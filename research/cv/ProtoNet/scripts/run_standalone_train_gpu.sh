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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh [DATA_PATH] [DEVICE_ID] [TRAIN_CLASS] [EPOCHS]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export DATA_PATH=$(get_real_path $1) # dataset path
export DEVICE_ID=$2 # device_id
export TRAIN_CLASS=$3 # train class, propose 20
export EPOCHS=$4 # num of epochs

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a directory"
exit 1
fi

python ../train.py --dataset_root=$DATA_PATH \
                --device_id=$DEVICE_ID --device_target="GPU" \
                --classes_per_it_tr=$TRAIN_CLASS \
                --experiment_root="./standalone_output" \
                --epochs=$EPOCHS > log 2>&1 &
