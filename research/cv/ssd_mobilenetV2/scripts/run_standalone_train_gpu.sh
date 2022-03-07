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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [DATASET] \
 [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
echo "for example: bash scripts/run_standalone_train_gpu.sh 0 500 coco"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 3 ] && [ $# != 5 ]
then
    echo "Usage: bash run_standalone_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [DATASET] \
 [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

# Before start 1pc train, first create mindrecord files.
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit
python train.py --only_create_dataset=True --dataset=$3 --run_platform="GPU"

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

DEVICE_ID=$1
EPOCH_SIZE=$2
DATASET=$3

if [ $# == 5 ]
then
    PRE_TRAINED=$4
    PRE_TRAINED_EPOCH_SIZE=$5
    
fi

rm -rf LOG$1
mkdir ./LOG$1
cp ./*.py ./LOG$1
cp -r ./src ./LOG$1
cp -r ./scripts ./LOG$1
cd ./LOG$1 || exit

echo "start training for device $1"
env > env.log
if [ $# == 3 ]
then
    python train.py  \
    --dataset=$DATASET \
    --device_id=$DEVICE_ID  \
    --run_platform="GPU" \
    --num_workers="16" \
    --use_float16="False" \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
fi

if [ $# == 5 ]
then
    python train.py  \
    --dataset=$DATASET \
    --device_id=$DEVICE_ID  \
    --pre_trained=$PRE_TRAINED \
    --run_platform="GPU" \
    --num_workers="16" \
    --use_float16="False" \
    --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &

fi

cd ../
