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

CURPATH="$(dirname "$0")"
# shellcheck source=/dev/null

if [ $# != 4 ]
then 
    echo "Usage: bash run_distribute_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [FP32_CKPT_PATH]"
    echo "PYTHON_PATH represents path to directory of 'train.py'."
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PYTHON_PATH=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)
DATASET_PATH=$(get_real_path $3)
FP32_CKPT_FILE=$(get_real_path $4)

if [ ! -d $PYTHON_PATH ]
then
    echo "error: PYTHON_PATH=$PYTHON_PATH is not a directory"
    exit 1
fi

if [ ! -f $CONFIG_FILE ]
then
    echo "error: CONFIG_FILE=$CONFIG_FILE is not a file"
    exit 1
fi

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
    exit 1
fi 

if [ ! -f $FP32_CKPT_FILE ]
then
    echo "error: FP32_CKPT_FILE=$FP32_CKPT_FILE is not a file"
    exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8

if [ -d "train_parallel" ];
then
    rm -rf ./train_parallel
fi
mkdir ./train_parallel
cp ${PYTHON_PATH}/*.py ./train_parallel
cp -r ${CURPATH}/../../src ./train_parallel
cd ./train_parallel || exit

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
       python train.py --config_path=$CONFIG_FILE --run_distribute=True --device_target="GPU" \
       --data_path=$DATASET_PATH --fp32_ckpt=$FP32_CKPT_FILE --output_path './output' &> log &
