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
echo "bash run_train_ascend.sh DEVICE_ID ACLIMDB_DIR GLOVE_DIR"
echo "for example: bash run_train_ascend.sh 0 ./aclimdb ./glove_dir"
echo "=============================================================================================================="

DEVICE_ID=$1
ACLIMDB_DIR=$2
GLOVE_DIR=$3

rm -rf train
mkdir train
cd train || exit
mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export DEVICE_ID=$DEVICE_ID

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../../config_ascend.yaml"

python ../../train.py  \
    --config_path=$CONFIG_FILE \
    --device_target="Ascend" \
    --aclimdb_path=$ACLIMDB_DIR \
    --glove_path=$GLOVE_DIR \
    --preprocess=true  \
    --preprocess_path=./preprocess > train.log 2>&1 &
