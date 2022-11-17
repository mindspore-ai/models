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

if [ $# != 4 ]; then
  echo "Usage: bash scripts/run_eval_onnx.sh [DATASET_NAME] [DATASET_PATH] [ONNX_MODEL] [DEVICE_TARGET]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_NAME=$1
DATASET_PATH=$(get_real_path $2)
ONNX_MODEL=$(get_real_path $3)
DEVICE_TARGET=${4:-"GPU"}

if [ ! -d $DATASET_PATH ]; then
  echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
  exit 1
fi

if [ ! -f $ONNX_MODEL ]; then
  echo "error: ONNX_MODEL=$ONNX_MODEL is not a file"
  exit 1
fi

python eval_onnx.py \
 --eval_dataset=$DATASET_NAME \
 --eval_dataset_path=$DATASET_PATH \
 --file_name=$ONNX_MODEL \
 --model_version=V2 \
 --device_target=$DEVICE_TARGET &> log.txt &
