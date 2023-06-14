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

echo "Please run the script as: "
echo "bash run_standalone_train_ascend.sh DATASET_PATH DEVICE_ID"
echo "for example: bash run_standalone_train_ascend.sh ~/DepthNet_dataset 0"
echo "After running the script, the network runs in the background, The log will be generated in train.log"

export RANK_ID=0
export DATASET_PATH=$1
DEVICE_ID=$2

cd ..
python train.py --data_url $DATASET_PATH --device_id $DEVICE_ID > train.log 2>&1 &
