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

if [ $# != 6 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [DEVICE_NUM] [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH] "
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_SIZE=$1
export LRPATH=$2
export GTPATH=$3
export VGGCKPT=$4
export VLRPATH=$5
export VGTPATH=$6

rm -rf ./train_parallel
mkdir ./train_parallel
cp -r ../src ./train_parallel
cp -r ../*.py ./train_parallel
cd ./train_parallel || exit

env > env.log
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
python train.py --run_distribute=1 --device_num=$RANK_SIZE \
      --train_LR_path=$LRPATH --train_GT_path=$GTPATH --vgg_ckpt=$VGGCKPT \
      --val_LR_path=$VLRPATH --val_GT_path=$VGTPATH --platform=GPU > log 2>&1 &
cd ..
