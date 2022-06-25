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
if [[ $# -eq 1 ]]; then
    echo "Usage: bash run_infer_onnx.sh [DEVICE_ID] 
    DEVICE_ID means device id, it can be set by environment variable DEVICE_ID.
    for example: bash run_infer_onnx.sh 0
    You can also run 'python eval_onnx.py --config_path=../onnx_infer_config.yaml' command to run the script.
    Please Check the file path in Default_config.yaml.    Make sure your related files paths right."
exit 1
fi

DEVICE_ID=$1
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../onnx_infer_config.yaml"
python ../eval_onnx.py  \
    --config_path=$CONFIG_FILE > onnx_infer.txt
