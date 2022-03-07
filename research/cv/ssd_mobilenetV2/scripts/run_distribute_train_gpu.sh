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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
echo "for example: bash scripts/run_distribute_train_gpu.sh 4 500 coco"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 3 ] && [ $# != 5 ]
then
    echo "Usage: bash scripts/run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [DATASET] \
 [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

# Before start distribute train, first create mindrecord files.
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit
python train.py --only_create_dataset=True --dataset=$3 --run_platform="GPU"

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_SIZE=$1
EPOCH_SIZE=$2
DATASET=$3

if [ $# == 5 ]
then
    PRE_TRAINED=$4
    PRE_TRAINED_EPOCH_SIZE=$5
fi

if [ $# == 3 ]
then
    mpirun -allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --distribute=True  \
    --dataset=$DATASET \
    --run_platform="GPU" \
    --lr="0.2" \
    --use_float16="False" \
    --num_workers="16" \
    --device_num=$RANK_SIZE  \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
fi

if [ $# == 5 ]
then
    mpirun -allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --distribute=True  \
    --dataset=$DATASET \
    --device_num=$RANK_SIZE  \
    --run_platform="GPU" \
    --use_float16="False" \
    --num_workers="16" \
    --pre_trained=$PRE_TRAINED \
    --lr="0.2" \
    --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &

fi

