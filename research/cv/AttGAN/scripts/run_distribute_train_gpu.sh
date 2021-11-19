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

if [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [EXPERIMENT_NAME] [DATA_PATH] [ATTR_PATH]"
    exit 1
fi

export RANK_SIZE=8
echo "After running the script, the network runs in the background. The log will be generated in train_parallel/log.txt"

experiment_name=$1
data_path=$2
attr_path=$3

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp *.sh ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python train.py \
      --img_size 128 \
      --shortcut_layers 1 \
      --inject_layers 1 \
      --experiment_name $experiment_name \
      --data_path $data_path \
      --attr_path $attr_path \
      --platform "GPU" \
      --run_distribute 1 > log 2>&1 &
cd ..
