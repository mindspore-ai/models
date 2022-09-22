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

if [ $# != 4 ]
then
    echo "Usage: bash run_infer_onnx.sh [ONNX_PATH] [DATA_PATH] [OUTPUT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ONNX_PATH=$(get_real_path $1)
DATA_PATH=$(get_real_path $2)
OUTPUT_PATH=$(get_real_path $3)

export ONNX=$ONNX_PATH
export DATASET=$DATA_PATH
export OUTPUT_PATH=$OUTPUT_PATH
export DEVICE_ID=$4

echo "start eval on DEVICE $DEVICE_ID"
echo "the results will saved in $OUTPUT_PATH"
python -u ../infer_onnx.py --onnx_path=$ONNX --data_path=$DATASET --device_id=$DEVICE_ID --output_path=$OUTPUT_PATH --device_target=GPU > eval_onnx_log 2>&1 &
