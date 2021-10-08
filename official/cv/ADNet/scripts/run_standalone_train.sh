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
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh /path/dataset DEVICE_ID"
echo "For example: bash run.sh dataset_path 3"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
execute_path=$(pwd)
echo ${execute_path}
DATASET=$1
DEVICE_ID=$2
echo "Start training for device $DEVICE_ID."
python3.7 -u train.py --dataset_path ${DATASET} --target_device ${DEVICE_ID} > log${DEVICE_ID} 2>&1 &
