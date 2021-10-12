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
if [ $# != 3 ]
then
    echo "Usage: sh run_distribution_train_gpu.sh [cifar10] [DEVICE_NUM] [TRAIN_DATASET_PATH]"
else

#
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# Parameters.
ulimit -u unlimited
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
export DATASET_NAME=$1
export DEVICE_NUM=$2
export TRAIN_DATASET_PATH=$(get_real_path $3)

# Base.
rm -rf ./train_parallel
mkdir ./train_parallel
cp -r $self_path/../src ./train_parallel
cp $self_path/../train.py ./train_parallel
cd ./train_parallel || exit

# Train.
mpirun --allow-run-as-root -n $DEVICE_NUM --output-filename log_output \
    python train.py --device_num=$DEVICE_NUM \
                    --device_target="GPU" --train_dataset_path=$TRAIN_DATASET_PATH \
                    --run_cloudbrain=False --run_distribute=True \
> log 2>&1 &
cd ..
fi
