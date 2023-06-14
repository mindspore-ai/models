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
if [ $# != 3 ]
then
    echo "Usage: bash run_standalone_train_for_gpu_resnext101.sh [DEVICE_ID] [DATA_PATH] [CONFIG_PATH]."
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ ! -d "$(get_real_path $2)" ]
then
    echo "error: DATA_PATH=$2 is not a directory"
    echo "Usage: bash run_standalone_train_for_gpu_resnext101.sh [DEVICE_ID] [DATA_PATH] [CONFIG_PATH]."
exit 1
fi

if [ ! -f "$(get_real_path $3)" ]
then
    echo "error: CONFIG_PATH=$3 is not a file"
    echo "Usage: bash run_standalone_train_for_gpu_resnext101.sh [DEVICE_ID] [DATA_PATH] [CONFIG_PATH]."
exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
DATA_DIR=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $3)
TRAIN_DATA_PATH="${DATA_DIR}/train"
EVAL_DATA_PATH="${DATA_DIR}/val"

python train.py  \
    --data_path=${TRAIN_DATA_PATH} \
    --eval_data_path=${EVAL_DATA_PATH} \
    --config_path=${CONFIG_PATH} > train.log 2>&1 &
