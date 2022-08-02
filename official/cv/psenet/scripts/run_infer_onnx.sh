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

if [ $# -ne 2 ]; then
    echo "Usage: bash run_infer_onnx.sh [ONNX_PATH] [TEST_DATASET_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
onnx_path=$(get_real_path $1)
data_path=$(get_real_path $2)

echo "ONNX name: "$onnx_path
echo "dataset path: "$data_path

function infer()
{
    python ./infer_psenet_onnx.py --onnx_path=$onnx_path \
                                  --TEST_ROOT_DIR=$data_path &> infer_onnx.log
}

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
