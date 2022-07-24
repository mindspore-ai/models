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

if [ $# -ne 1 ];
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_onnx_verify.sh"
  echo "For example: bash run_onnx_verify.sh"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
  exit 2
fi

set -e

DEVICE_ID=$1

export DEVICE_ID=$DEVICE_ID

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf verify_onnx/
mkdir verify_onnx
cd ./verify_onnx
mkdir src
cd ../
cp ./default_paras.yaml ./verify_onnx
cp ./verify_onnx.py ./verify_onnx
cp ./src/*.py ./verify_onnx/src
cp ./actornet.onnx ./verify_onnx
cd ./verify_onnx

env > env0.log
echo "Verify begin."
python verify_onnx.py > ./verify_onnx.log 2>&1 &

if [ $? -eq 0 ];then
    echo "evaling success"
else
    echo "evaling failed"
    exit 2
fi
echo "finish"
cd ../
