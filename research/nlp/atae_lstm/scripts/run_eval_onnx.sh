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
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_eval_onnx.sh [ONNX_CONFIG] [ONNX_CKPT] [DATA_DIR] [DEVICE]"
  echo "for example:"
  echo "bash run_eval_onnx.sh \
    ./onnx_config.yaml \
    ./ATAE_LSTM.onnx \
    ./mindrecord_data \
    Ascend"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
fi

ONNX_CONFIG=$1
ONNX_CKPT=$2
DATA_DIR=$3
DEVICE=$4

python eval_onnx.py \
    --config="$ONNX_CONFIG" \
    --onnx_ckpt="$ONNX_CKPT" \
    --data_url="$DATA_DIR" \
    --device="$DEVICE" > eval_log.log 2>&1 &