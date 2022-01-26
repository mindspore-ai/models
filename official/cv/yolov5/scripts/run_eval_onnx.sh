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

if [ $# -lt 2 ]
then
    usage="Usage: bash run_eval_onnx.sh \
<DATA_DIR> <ONNX_MODEL_PATH> \
[<DEVICE_TARGET>]"
    echo "$usage"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_DIR=$(get_real_path $1)
ONNX_MODEL_PATH=$(get_real_path $2)
DEVICE_TARGET=${3:-"GPU"}

python eval_onnx.py \
    --data_dir $DATA_DIR \
    --file_name $ONNX_MODEL_PATH \
    --device_target $DEVICE_TARGET \
    --per_batch_size 1 &> eval.log &
