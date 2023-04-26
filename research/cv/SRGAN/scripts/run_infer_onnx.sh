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
if [ $# != 4 ]
then
    echo "Usage: bash run_eval_onnx.sh [TEST_LR_PATH] [TEST_GT_PATH] [ONNX_PATH] [DEVICE_TARGET]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

TEST_LR_PATH=$(get_real_path $1)
TEST_GT_PATH=$(get_real_path $2)
ONNX_PATH=$(get_real_path $3)
DEVICE_TARGET=$4

echo "TEST_LR_PATH: "$TEST_LR_PATH
echo "TEST_GT_PATH: "$TEST_GT_PATH
echo "ONNX_PATH: "$ONNX_PATH
echo "DEVICE_TARGET: "$DEVICE_TARGET

rm -rf ./infer
mkdir ./infer
cp -r ./src ./infer
cp -r ./*.py ./infer
cd ./infer || exit

function infer()
{
    python ./infer_onnx.py --test_LR_path=$TEST_LR_PATH \
                           --test_GT_path=$TEST_GT_PATH\
                           --onnx_path=$ONNX_PATH \
                           --device_target=$DEVICE_TARGET &> log
}
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi