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
  echo "Usage: bash run_gpu_distribute_train.sh  [TRAIN_DATA_DIR] [DEVICE_NUM]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

MY_PATH=$(get_real_path $1)
DEVICE_NUM=$2

if [ ! -d $MY_PATH ]; then
  echo "error: TRAIN_DATA_DIR=$MY_PATH is not a directory"
  exit 1
fi

if [ -d "train_parallel" ]; then
    rm -rf ./train_parallel
fi
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit

env >env.log

nohup mpirun --allow-run-as-root -n $DEVICE_NUM \
python train.py \
      --device_target GPU \
      --device_num $DEVICE_NUM \
      --batch_size 16 \
      --lr 2.4e-4 \
      --scale 2 \
      --task_id 0 \
      --dir_data $MY_PATH \
      --epochs 300 \
      --patch_size 48 > nohup_dist_train.log 2>&1 &