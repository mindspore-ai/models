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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_onnx_eval.sh CKPT_FILES  DEVICE_TYPE "
echo "for example: bash scripts/run_onnx_eval.sh /path/a.onnx GPU "
echo "=============================================================================================================="

if [ $# -lt 2 ]
then
    echo "Usage: bash scripts/run_onnx_eval.sh [CKPT_FILES] [DEVICE_TYPE]"
    exit 1
fi

CKPT_FILES=$1
DEVICE_TYPE=$2

python eval_onnx.py --ckpt_path=$CKPT_FILES \
    --target_device=$DEVICE_TYPE > output.eval_onnx.log 2>&1 &
