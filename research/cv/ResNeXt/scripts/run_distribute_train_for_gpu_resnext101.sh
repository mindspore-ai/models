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
    echo "Usage: bash run_distribute_train_for_gpu_resnext101.sh [DATA_PATH] [CONFIG_PATH]."
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ ! -d "$(get_real_path $1)" ]
then
    echo "error: DATA_PATH=$1 is not a directory"
    echo "Usage: bash run_distribute_train_for_gpu_resnext101.sh [DATA_PATH] [CONFIG_PATH]."
exit 1
fi

if [ ! -f "$(get_real_path $2)" ]
then
    echo "error: CONFIG_PATH=$2 is not a file"
    echo "Usage: bash run_distribute_train_for_gpu_resnext101.sh [DATA_PATH] [CONFIG_PATH]."
exit 1
fi

DATA_PATH=$(get_real_path $1)
CONFIG_PATH=$(get_real_path $2)

TRAIN_DATA_PATH="${DATA_PATH}/train"
VALID_DATA_PATH="${DATA_PATH}/val"

rm -rf logs
mkdir logs
cp -r ./train.py ./resnext101_config.yaml ./src ./logs
cd ./logs
echo "start training"
mpirun --allow-run-as-root -n 8 python ./train.py --data_path=${TRAIN_DATA_PATH} --eval_data_path=${VALID_DATA_PATH} --run_distribute=True --config_path=${CONFIG_PATH} > train.log 2>&1 &
