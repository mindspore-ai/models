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

set -e
if [ $# -ne 5 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bbash run_onnx_eval.sh DATA_PATH TEST_PATH ONNX_PATH CONFIG_PATH DEVICE_ID"
    echo "For example: bash run_onnx_eval.sh ./data ./test ./onnx default_config.ymal 0"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi

DATA_PATH=$1
TEST_PATH=$2
ONNX_PATH=$3
CONFIG_PATH=$4
DEVICE_ID=$5

export DATA_PATH=${DATA_PATH}
export TEST_PATH=${TEST_PATH}
export ONNX_PATH=${ONNX_PATH}
export CONFIG_PATH=${CONFIG_PATH}
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

if [ ! -d "$DATA_PATH" ]; then
        echo "dataset does not exit"
        exit
fi

echo "eval_onnx begin."
cd ../
nohup python eval_onnx.py > eval_onnx.log 2>&1 \
--batch_size=1 \
--data_path=$DATA_PATH \
--test_path=$TEST_PATH \
--onnx_path=$ONNX_PATH \
--config_path=$CONFIG_PATH \
--device_id=$DEVICE_ID &
echo "eval_onnx background..."
