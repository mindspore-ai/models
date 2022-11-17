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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_single_train.sh DEVICE_ID MINDRECORD_DIR OUTPUT_PATH PRE_TRAINED_PATH"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: bash run_single_train.sh [DEVICE_ID] [MINDRECORD_DIR] \
[OUTPUT_PATH] [PRE_TRAINED_PATH](optional)"
    exit 1
fi

# Before start single train, first create mindrecord files.
# BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
# cd $BASE_PATH/../ || exit
# python train.py --only_create_dataset=True

echo "After running the script, the network runs in the background. The log will be generated in log.txt"

export DEVICE_ID=$1
MINDRECORD_DIR=$2
OUTPUT_PATH=$3
PRE_TRAINED_PATH=$4



echo "start training for device $1"
env > env.log
if [ $# == 3 ]
then
    python ../train.py  \
    --data_path=$MINDRECORD_DIR --output_path=$OUTPUT_PATH > log.txt 2>&1 &
fi

if [ $# == 4 ]
then
    python train.py  \
    --data_path=$MINDRECORD_DIR \
    --output_path=$OUTPUT_PATH \
    --pre_trained_epoch_size=$PRE_TRAINED_PATH > log.txt 2>&1 &
fi


