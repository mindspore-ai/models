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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh DATA_PATH RANK_SIZE CONFIG_PATH"
echo "For example: bash run_distribute_train.sh /path/dataset 8 ../config/config_resnet50_gpu.yaml"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
if [ $# != 4 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [RANK_SIZE] [CONFIG_PATH]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

set -e
DEVICE_NUM=$3
DATA_PATH=$(get_real_path $1)
EVAL_DATA_PATH=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $4)
export DATA_PATH=${DATA_PATH}
export DEVICE_NUM=$3
export RANK_SIZE=$3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf ./train_parallel
mkdir ./train_parallel
cd ./train_parallel
mkdir src
cd ../
cp *.py ./train_parallel
cp src/*.py ./train_parallel/src
cd ./train_parallel
env > env.log
echo "start training"
    mpirun -n $3 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
           python3 train.py --data_url=$DATA_PATH --isModelArts=False --run_distribute=True \
           --device_target="GPU" --config_path=$CONFIG_PATH --eval_data_url=$EVAL_DATA_PATH --device_num $3 > train.log 2>&1 &

