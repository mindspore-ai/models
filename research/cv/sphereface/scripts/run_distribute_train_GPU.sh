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


echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_GPU.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]"
echo "For example: bash run_distribute_train_GPU.sh 8 0,1,2,3,4,5,6,7"
echo "=============================================================================================================="
set -e
if [ $# -lt 1 ]
then
    echo "Usage: sh run_distribute_train_GPU.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]"
    exit 1
fi
export DEVICE_NUM=$1
export RANK_SIZE=$1
export CUDA_VISIBLE_DEVICES=$2
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

rm -rf GPU_distributed
mkdir GPU_distributed
cd ./GPU_distributed
mkdir src
cd ../
cp ./*.py ./GPU_distributed
cp ./*.yaml ./GPU_distributed
cp -r ./src ./GPU_distributed
cd ./GPU_distributed

echo "start training for GPU"
env > env.log
mpirun -n $1 --allow-run-as-root python train.py \
      --per_batch_size=64 \
      --T_max=23 \
      --max_epoch=25 \
      --ckpt_interval=2500 \
      --is_distributed=1 \
      --device_target='GPU' > output.log 2>&1 &
echo "finish"
cd ../
