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

[ $# -ne 2 ] && {
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_PATH]"
    exit 1
}

export RANK_SIZE=1
export DEVICE_ID=$1
DATA_DIR=$2
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config_gpu.yaml"

rm -rf train_standalone
mkdir ./train_standalone
cd ./train_standalone || exit
echo  "start training for device id $DEVICE_ID"
env > env.log
python -u ../train.py --config_path=$CONFIG_FILE \
    --dataset_path=$DATA_DIR --platform=GPU > log.txt 2>&1 &
cd ../
