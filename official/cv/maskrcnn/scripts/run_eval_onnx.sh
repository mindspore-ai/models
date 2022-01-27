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

if [ $# != 2 ]
then
    echo "Usage: bash run_onnx_eval.sh <ONNX_MODEL_PATH> <DATA_PATH> [<DEVICE_TARGET>]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
ONNX_MODEL_PATH=$(get_real_path $1)
DATA_PATH=$(get_real_path $2)
DEVICE_TARGET=${3:-"GPU"}

mkdir eval

python eval_onnx.py --file_name=$ONNX_MODEL_PATH --coco_root=$DATA_PATH --device_target=$DEVICE_TARGET &> eval/log_eval_onnx.txt &
