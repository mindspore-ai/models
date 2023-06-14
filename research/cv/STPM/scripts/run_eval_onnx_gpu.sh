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
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval_onnx_gpu.sh DATASET_PATH ONNX_PATH CATEGORY DEVICE_ID"
    echo "For example: bash run_eval_onnx_gpu.sh /path/dataset /path/onnx category 0"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi
set -e

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $1)
ONNX_PATH=$(get_real_path $2)
category=$3
device_id=$4

eval_path=eval_onnx_$category
if [ -d $eval_path ];
then
    rm -rf ./$eval_path
fi
mkdir ./$eval_path
cd ./$eval_path
env > env0.log
echo "[INFO] start eval dataset $category with device_id: $device_id"
python ../../eval_onnx.py \
--data_url $DATA_PATH  \
--onnx_path $ONNX_PATH \
--category $category \
--device_id $device_id \
> eval_onnx.log 2>&1

if [ $? -eq 0 ];then
    echo "[INFO] eval success"
else
    echo "[ERROR] eval failed"
    exit 2
fi
echo "[INFO] finish"
cd ../
