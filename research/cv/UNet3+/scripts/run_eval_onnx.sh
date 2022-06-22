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
if [ $# != 3 ];then
  echo "Usage:bash scripts/eval_onnx.sh [DATASET_PATH] [ONNX_MODEL] [DEVICE_TARGET]"
  exit 1
fi


get_real_path(){
  if [ "${1:0:1}" == "/" ];then
    echo "$1"
  else
    realpath -m "$PWD/$1"
  fi
}

DATASET_PATH=$(get_real_path "$1")
ONNX_MODEL=$(get_real_path "$2")
DEVICE_TARGET=${3}

if [ ! -d "$DATASET_PATH" ];then
  echo "error:DATASET_PATH=${DATASET_PATH} is not a directory"
  exit 1
fi

if [ ! -f "$ONNX_MODEL" ];then
  echo "error:ONNX_MODEL=${ONNX_MODEL} is not a file"
  exit 1
fi

python ../eval_onnx.py \
  --val_data_path="$DATASET_PATH" \
  --file_name="$ONNX_MODEL" \
  --device_target="$DEVICE_TARGET" > eval_onnx.log 2>&1 &
