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
echo "bash run_eval_ascend.sh DEVICE_ID PREPROCESS_DIR CKPT_FILE"
echo "for example: bash run_eval_ascend.sh 0 ./preprocess lstm_best_acc.ckpt"
echo "=============================================================================================================="

DEVICE_ID=$1
PREPROCESS_DIR=$2
CKPT_FILE=$3

rm -rf eval
mkdir -p eval
cd eval || exit
mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export DEVICE_ID=$DEVICE_ID

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../../config_ascend.yaml"

python ../../eval.py  \
    --config_path=$CONFIG_FILE \
    --device_target="Ascend" \
    --preprocess=false \
    --preprocess_path=$PREPROCESS_DIR \
    --ckpt_file=$CKPT_FILE > eval.log 2>&1 &
