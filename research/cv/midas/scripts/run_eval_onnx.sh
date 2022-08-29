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

if [ $# -ne 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval_onnx.sh DEVICE_ID DATA_NAME device_target"
    echo "For example: bash run_eval_onnx.sh 0 all GPU"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi

export DEVICE_ID=$1
export DATA_NAME=$2
export device_target=$3

python -u ../midas_eval_onnx.py --device_id=$DEVICE_ID --data_name=$DATA_NAME --device_target=$device_target > eval_onnx.txt 2>&1 &
