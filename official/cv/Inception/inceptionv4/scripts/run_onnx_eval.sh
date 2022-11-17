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
echo "bash scripts/run_onnx_eval.sh [DATA_PATH] [DATASET_TYPE] [DEVICE_TYPE] [FILE_TYPE] [ONNX_PATH]"
echo "[DATASET_TYPE] should be in [imagenet, cifar10]"
echo "for example: bash scripts/run_onnx_eval.sh /path/ImageNet2012/validation imagenet GPU ONNX /path/inceptionv4.onnx "
echo "=============================================================================================================="

if [ $# != 5 ]
then
    echo "bash scripts/run_onnx_eval.sh [DATA_PATH] [DATASET_TYPE] [DEVICE_TYPE] [FILE_TYPE] [ONNX_PATH]"
exit 1
fi

DATA_PATH=$1
DATASET_TYPE=$2
DEVICE_TYPE=$3
FILE_TYPE=$4
ONNX_PATH=$5

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
config_path="${BASE_PATH}/../default_config_gpu.yaml"

echo "start evaluation"

python eval_onnx.py \
    --config_path=$config_path \
    --data_path=$DATA_PATH \
    --ds_type=$DATASET_TYPE \
    --device_target=$DEVICE_TYPE \
    --file_format=$FILE_TYPE \
    --file_name=$ONNX_PATH &> output.eval_onnx.log &
