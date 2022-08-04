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
  echo "Usage:bash scripts/run_eval_onnx.sh [DEVICE_TARGET] [TASK_NAME] [DATA_FILE_PATH] [ONNX_MODEL]"
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ];then
    echo "$1"
  else
    realpath -m "$PWD/$1"
  fi
}

DEVICE_TARGET=${1}
TASK_NAME=${2}
DATA_FILE_PATH=$(get_real_path "$3")
ONNX_MODEL=$(get_real_path "$4")

if [ ! -f "$DATA_FILE_PATH" ];then
  echo "error:DATA_FILE_PATH=${DATA_FILE_PATH} is not a file"
  exit 1
fi

if [ ! -f "$ONNX_MODEL" ];then
  echo "error:ONNX_MODEL=${ONNX_MODEL} is not a file"
  exit 1
fi

python3 eval_onnx.py \
         --device_target ${DEVICE_TARGET}  \
         --device_id 0 \
         --eval_batch_size 100 \
         --task_name ${TASK_NAME} \
         --eval_data_file_path ${DATA_FILE_PATH} \
         --eval_ckpt_path ${ONNX_MODEL} > eval_onnx.log 2>&1 &

