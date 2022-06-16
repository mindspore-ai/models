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
if [ $# -lt 2 ]
then
    echo "Usage: bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CONFIG_PATH]"
exit 1
fi

export DEVICE_ID=$1
CONFIG_PATH=$2

if [ ! -d log ] 
then
    mkdir -p ./log
fi

python train.py  \
--config $CONFIG_PATH \
--device_id $DEVICE_ID \
--save_every 20 2>&1 | tee log/train_"$(date +%F_%H-%M-%S)".log
