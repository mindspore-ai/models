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
if [ $# != 4 ];then
  echo "Usage:bash scripts/eval_onnx.sh [DATASET_PATH] [ONNX_MODEL] [DEVICE_TARGET] [CONFIG_PATH]"
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
CONFIG_PATH=$(get_real_path "$4")

if [ ! -d "$DATASET_PATH" ];then
  echo "error:DATASET_PATH=${DATASET_PATH} is not a directory"
  exit 1
fi

if [ ! -f "$ONNX_MODEL" ];then
  echo "error:ONNX_MODEL=${ONNX_MODEL} is not a file"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ];then
  echo "error:CONFIG_PATH=${CONFIG_PATH} is not a file"
  exit 1
fi

 python infer_unet_onnx.py \
  --config_path="$CONFIG_PATH" \
  --file_name="$ONNX_MODEL" \
  --data_path="$DATASET_PATH" \
  --device_target="$DEVICE_TARGET" > eval_onnx.log 2>&1 &
