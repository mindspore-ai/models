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

if [ $# != 2 ]; then
  echo "Usage: bash run_distribute_train_gpu.sh [RANK_SIZE] [DATASET_PATH]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ulimit -u unlimited
export RANK_SIZE=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATASET_PATH=$(get_real_path $2)

if [ ! -d $DATASET_PATH ]; then
  echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
  exit 1
fi

if [ -d "distribute_train" ]; then
  rm -rf ./distribute_train
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
mkdir ./distribute_train
cp $BASEPATH/../*.py ./distribute_train
cp $BASEPATH/*.sh ./distribute_train
cp -r $BASEPATH/../src ./distribute_train
cd ./distribute_train || exit


mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python train.py  \
  --run_distribute=True \
  --dataset_path=$DATASET_PATH \
  --device_target=GPU > log.txt 2>&1 &
cd ..