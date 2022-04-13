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

if [ $# != 2 ]
then
    echo "===================================================================================================="
    echo "Please run the script as: "
    echo "bash script/run_standalone_train_gpu.sh CONFIG_PATH DATASET_PATH"
    echo "for example: bash script/run_standalone_train_gpu.sh /path/to/gpu_config.yaml /path/to/dataset/images"
    echo "===================================================================================================="
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD/$1"
  fi
}

CONFIG_PATH=$(get_real_path "$1")
DATASET_PATH=$(get_real_path "$2")

if [ -d logs ]
then
  rm -rf logs
fi

mkdir logs

python ./train.py --config_path "$CONFIG_PATH" --train_data_path "$DATASET_PATH" > ./logs/train.log 2>&1 &
