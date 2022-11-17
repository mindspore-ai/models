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
echo "bash scripts/run_onnx_eval.sh DATA_PATH DATASET_TYPE DEVICE_TYPE ONNX_MODEL_PATH"
echo "for example: bash scripts/run_onnx_eval.sh /path/ImageNet2012/validation imagenet2012 GPU /path/a.onnx "
echo "=============================================================================================================="

DATA_PATH=$1
DATASET_TYPE=$2
DEVICE_TYPE=$3
ONNX_MODEL_PATH=$4

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

config_path=$(get_real_path "./${DATASET_TYPE}_config.yaml")
echo "config path is : ${config_path}"

python eval_onnx.py \
    --config_path=$config_path \
    --data_dir=$DATA_PATH \
    --dataset=$DATASET_TYPE \
    --device_target=$DEVICE_TYPE \
    --file_name=$ONNX_MODEL_PATH > output.eval_onnx.log 2>&1 &
