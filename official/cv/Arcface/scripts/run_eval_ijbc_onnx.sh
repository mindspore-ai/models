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

if [ $# -ne 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval_ijbc_onnx.sh EVAL_PATH ONNX_BS_PATH ONNX_RS_PATH TARGET"
    echo "For example: bash run_eval_ijbc_onnx.sh path/evalset path/onnx_bs path/onnx_rs IJBC"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi

EVAL_PATH=$1
ONNX_BS_PATH=$2
ONNX_RS_PATH=$3
TARGET=$4

nohup python ../eval_ijbc_onnx.py \
--image-path "$EVAL_PATH" \
--onnx_bs_dir "$ONNX_BS_PATH" \
--onnx_rs_dir "$ONNX_RS_PATH" \
--batch-size 128 \
--result-dir onnx_ms1mv2_arcface_r100 \
--job onnx_ms1mv2_arcface_r100 \
--target $TARGET \
> eval_onnx_$TARGET.log 2>&1 &
