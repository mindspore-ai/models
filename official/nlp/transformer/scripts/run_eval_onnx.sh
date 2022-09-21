#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
if [ $# -lt 2 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_onnx.sh <ONNX_MODEL> <MINDRECORD_DATA> [<CONFIG_PATH>] [<DEVICE_TARGET>] [<DEVICE_ID>]"
echo "=============================================================================================================="
exit 1;
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ONNX_MODEL=$(get_real_path $1)
MINDRECORD_DATA=$(get_real_path $2)
CONFIG_PATH=${3:-"./default_config_large.yaml"}
CONFIG_PATH=$(get_real_path $CONFIG_PATH)
DEVICE_TARGET=${4:-"GPU"}
DEVICE_ID=${5:-0}


python eval_onnx.py  \
    --config_path=$CONFIG_PATH \
    --device_target=$DEVICE_TARGET \
    --device_id=$DEVICE_ID \
    --data_file=$MINDRECORD_DATA \
    --file_name=$ONNX_MODEL > log_eval.txt 2>&1 &
