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
echo "bash run_eval_cpu.sh ACLIMDB_DIR GLOVE_DIR CKPT_FILE"
echo "for example: bash run_eval_cpu.sh ./aclimdb ./glove_dir lstm_best_acc.ckpt"
echo "=============================================================================================================="

ACLIMDB_DIR=$1
GLOVE_DIR=$2
CKPT_FILE=$3

mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config.yaml"

python ../eval.py  \
    --config_path=$CONFIG_FILE \
    --device_target="CPU" \
    --aclimdb_path=$ACLIMDB_DIR \
    --glove_path=$GLOVE_DIR \
    --preprocess=false \
    --preprocess_path=./preprocess \
    --ckpt_file=$CKPT_FILE > eval.log 2>&1 &
