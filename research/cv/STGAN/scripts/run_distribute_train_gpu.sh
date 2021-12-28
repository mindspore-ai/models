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

if [ $# != 2 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [EXPERIMENT_NAME] [DATA_PATH]"
exit 1
fi


export DEVICE_NUM=8
export RANK_SIZE=8

rm -rf ./train_parallel
mkdir ./train_parallel
cp ./*.py ./train_parallel
cp -r ./src ./train_parallel
cp -r ./scripts ./train_parallel
cd ./train_parallel || exit

export EXPERIMENT_NAME=$1
export DATA_PATH=$2

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python train.py \
      --dataroot=$DATA_PATH \
      --experiment_name=$EXPERIMENT_NAME \
      --device_num ${DEVICE_NUM} \
      --platform="GPU" > log 2>&1 &
cd ..
