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
if [ $# != 6 ] && [ $# != 7 ]; then
    echo "Usage: bash scripts/run_infer_onnx.sh [INPUT_DIR] [INPUT_NAME] [DEVICE_TARGET] [ONNX_DIR] [MODEL_DIR] [INFER_OUTPUT] [DEVICE_ID]"
    echo "DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
fi
input_dir=$1
input_name=$2
device_target=$3
device_id=0
onnx_dir=$4
model_dir=$5
infer_output=$6

if [ $# == 7 ]; then
    device_id=$7
fi
nohup python -u infer_onnx.py --input_dir=$input_dir --input_name=$input_name\
                              --device_target=$device_target\
                              --onnx_dir=$onnx_dir --model_dir=$model_dir\
                              --infer_output=$infer_output\
                              --device_id=$device_id\
                              > log 2>&1 &
