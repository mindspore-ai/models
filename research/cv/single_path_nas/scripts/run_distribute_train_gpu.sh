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


if [ $# != 0 ] && [ $# != 1 ]
then
  echo "Usage: bash run_distribute_train_gpu.sh [TRAIN_DATASET](optional)"
  exit 1
fi

if [ $# == 1 ] && [ ! -d $1 ]
then
  echo "error: TRAIN_DATASET=$1 is not a directory"
  exit 1
fi

ulimit -u unlimited

rm -rf ./train_parallel
mkdir ./train_parallel
cp ./*.py ./train_parallel
cp -r ./src ./train_parallel
cd ./train_parallel || exit
env > env.log

if [ $# == 0 ]
then
  mpirun -n 8 \
    --allow-run-as-root \
    --output-filename 'log_output' \
    --merge-stderr-to-stdout \
    python ./train.py \
      --use_gpu_distributed=1 \
      --device_target='GPU' \
      --lr_init=1.5 > log.txt 2>&1 &
fi

if [ $# == 1 ]
then
  mpirun -n 8 \
    --allow-run-as-root \
    --output-filename 'log_output' \
    --merge-stderr-to-stdout \
    python ./train.py \
      --use_gpu_distributed=1 \
      --device_target='GPU' \
      --data_path="$1" \
      --lr_init=1.5 > log.txt 2>&1 &
fi
