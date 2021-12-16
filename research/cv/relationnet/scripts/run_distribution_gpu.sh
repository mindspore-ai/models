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

if [ $# != 2 ]; then
  echo "Usage: bash run_distribution_gpu.sh [DATASET_PATH] [DEVICE_NUM]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)

if [ ! -d $DATASET_PATH ]; then
  echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
  exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=$2
export RANK_SIZE=$2

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp *.sh ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit

if [ $# == 2 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
           python train.py --run_distribute=True --device_num=$DEVICE_NUM --device_target="GPU" \
           --data_path=$DATASET_PATH &> log &
fi