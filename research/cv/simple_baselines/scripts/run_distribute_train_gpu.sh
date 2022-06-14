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

if [ $# -ne 1 ]; then
    echo "Please run the script as: "
    echo "bash scripts/run_distribute_train_gpu.sh [RANK_SIZE]"
    echo "For example: bash scripts/run_distribute_train_gpu.sh 8"
    echo "It is better to use the absolute path."
    echo "========================================================================"
    exit 1
fi

export RANK_SIZE=$1

rm -rf ./train_parallel
mkdir ./train_parallel
cp ./*.py ./train_parallel
cp -r ./src ./train_parallel
cd ./train_parallel

echo "start training on GPU $RANK_SIZE devices"
env > env.log

mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
python train.py \
    --device_target="GPU" \
    --is_model_arts=False \
    --run_distribute=True > train.log 2>&1 &
cd ..
