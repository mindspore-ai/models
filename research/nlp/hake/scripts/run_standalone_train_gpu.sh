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

if [ $# != 3 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_standalone_train_gpu.sh DEVICE_ID DATA_PATH SAVE_PATH"
echo "for example: bash scripts/run_standalone_train_gpu.sh 3 data/wn18rr wn18rr/"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

DEVICE_ID=$1
DATA_PATH=$2
SAVA_PATH=$3

if [ ! -d "$SAVA_PATH" ]; then
        mkdir $SAVA_PATH
fi

python -u train.py  \
    --do_distribute="False" \
    --device_id=$DEVICE_ID \
    --data_path=$DATA_PATH \
    --save_path=$SAVA_PATH \
    > $SAVA_PATH/train_standalone.log 2>&1 &