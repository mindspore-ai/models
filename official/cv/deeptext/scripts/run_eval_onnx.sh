#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

if [ $# -lt 4 ]
then 
    echo "Usage: bash run_eval_onnx.sh [IMGS_PATH] \
[ANNOS_PATH] [ONNX_MODEL] [MINDRECORD_DIR] \
[<DEVICE_TARGET>]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

IMGS_PATH=$(get_real_path $1)
ANNOS_PATH=$(get_real_path $2)
ONNX_MODEL=$(get_real_path $3)
MINDRECORD_DIR=$(get_real_path $4)
DEVICE_TARGET=${5:-"GPU"}

python eval_onnx.py \
  --device_target=$DEVICE_TARGET \
  --export_device_target=$DEVICE_TARGET \
  --imgs_path=$IMGS_PATH \
  --annos_path=$ANNOS_PATH \
  --file_name=$ONNX_MODEL \
  --mindrecord_dir=$MINDRECORD_DIR &> log &
