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

if [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_onnx_eval_gpu.sh DATA_PATH DEVICE_ID ONNX_MODEL_PATH"
    echo "for example: bash scripts/run_onnx_eval_gpu.sh /path/ImageNet2012/validation 0 /path/a.onnx "
    echo "=============================================================================================================="
exit 1
fi

DATA_PATH=$1
DEVICE_ID=$2
ONNX_MODEL_PATH=$3

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

mpirun -n 1 python eval_onnx.py \
    --dataset_path=$DATA_PATH \
    --device_target="GPU" \
    --onnx_path=$ONNX_MODEL_PATH > eval_onnx.log 2>&1 &
