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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DEVICE_ID CKPT_PATH"
echo "For example: bash scripts/run_eval_onnx.sh ./DIV2K_config.yaml  2  DIV2K path output_path  pre_trained_model_path  ONNX"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

if [ $# != 6 ]
then
    echo "Usage:  bash scripts/run_eval_onnx.sh [config_path]  [scale]  [data_path] [output_path]  [pre_trained_model_path]  [eval_type]"
exit 1
fi

export args=${*:1}
python eval_onnx.py --config_path $1 --scale $2 --data_path $3 --output_path $4 --pre_trained $5 --eval_type $6 > eval_onnx.log 2>&1 &
