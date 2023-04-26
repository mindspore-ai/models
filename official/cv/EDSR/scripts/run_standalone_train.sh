#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: bash scripts/run_standalone_train.sh [SCALE] [DATA_PATH] [OUTPUT_PATH] [PRETRAIN_PATH](optional)"
exit 1
fi

SCALE=$1
DATA_PATH=$2
OUTPUT_PATH=$3

if [ $# == 4 ]
then
    PRETRAIN_PATH=$4
fi

if [ $SCALE == 2 ]
then
    python train.py --batch_size 16 --config_path DIV2K_config.yaml --scale $SCALE \
        --data_path $DATA_PATH --output_path $OUTPUT_PATH > train.log 2>&1 &
else
    python train.py --batch_size 16 --config_path DIV2K_config.yaml --scale $SCALE \
        --data_path $DATA_PATH --output_path $OUTPUT_PATH --pre_trained $PRETRAIN_PATH > train.log 2>&1 &
fi