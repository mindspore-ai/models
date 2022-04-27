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
echo "bash run_single_train.sh DEVICE_ID CONFIG_PATH"
echo "for example: sh run_single_train.sh 0 home/hed/config/default_config_910.yaml"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 2 ]
then
    echo "Usage: bash run_single_train.sh [DEVICE_ID] [CONFIG_PATH]"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export DEVICE_ID=$1
CONFIG_PATH=$2

rm -rf LOG$1
mkdir ./LOG$1
cp ./*.py ./LOG$1
cp -r ./src ./LOG$1
cp ./config/*.yaml ./LOG$1
cd ./LOG$1 || exit
echo "start training for device $1"
env > env.log
python train.py  \
--config_path=$CONFIG_PATH \
--device_id=$DEVICE_ID > log.txt 2>&1 &
cd ../
