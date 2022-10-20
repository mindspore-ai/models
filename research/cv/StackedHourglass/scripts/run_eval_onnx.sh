#!/usr/bin/env bash

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
  echo "Usage: ./scripts/run_eval_onnx.sh [onnx path] [annot path] [image path] [device target]"
  exit 1
fi
TARGET="./Eval_onnx"

#set -e
rm -rf $TARGET
mkdir $TARGET

ONNX_PATH=$1
ANNOT_PATH=$2
IMAGES_PATH=$3
DEVICE_TARGET=$4

python eval_onnx.py \
    --onnx_file ${ONNX_PATH} \
    --annot_dir ${ANNOT_PATH} \
    --img_dir ${IMAGES_PATH} \
    --device_target ${DEVICE_TARGET} > Eval_onnx/result.txt 2> Eval_onnx/err.txt
    