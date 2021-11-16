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
echo "bash run_standalone_train_gpu.sh"
echo "For example: bash run_standalone_train_gpu.sh MINDRECORD_PATH CONFIG_PATH DEVICE_ID"
echo "=============================================================================================================="
set -e
if [ $# != 3 ]
then
  echo "Usage: bash run_standalone_train_gpu.sh MINDRECORD_PATH CONFIG_PATH DEVICE_ID"
exit 1
fi

MINDRECORD_PATH=$1
CONFIG_PATH=$2
DEVICE_ID=$3


export MINDRECORD_PATH=${MINDRECORD_PATH}
export CONFIG_PATH=${CONFIG_PATH}
export DEVICE_ID=${DEVICE_ID}

echo "Standalone train begin."
cd ../
nohup python train.py > train_standalone.log 2>&1 \
--mindrecord_path=$MINDRECORD_PATH \
--config_path=$CONFIG_PATH \
--device_id=$DEVICE_ID \
--run_distribute=False &

echo "Training background..."
