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
  echo "Usage: bash run_standalone_train_gpu.sh [TRAIN_DATASET](optional)"
  exit 1
fi

if [ $# == 1 ] && [ ! -d $1 ]
then
  echo "error: TRAIN_DATASET=$1 is not a directory"
  exit 1
fi

ulimit -u unlimited

rm -rf ./train_standalone
mkdir ./train_standalone
cp ./*.py ./train_standalone
cp -r ./src ./train_standalone
cd ./train_standalone || exit
env > env.log

if [ $# == 0 ]
then
  python train.py --device_target='GPU' --lr_init=0.26 > log.txt 2>&1 &
fi

if [ $# == 1 ]
then
  python train.py --device_target='GPU' --data_path="$1" --lr_init=0.26 > log.txt 2>&1 &
fi
