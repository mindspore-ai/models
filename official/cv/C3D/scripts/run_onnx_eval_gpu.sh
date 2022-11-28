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

if [ $# -lt 2 ]; then
    echo "Usage: bash run_onnx_eval_gpu.sh  [ONNX_PATH] [CONFIG_PATH]
    ONNX_PATH is saved model onnx file path."
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
CONFIG_PATH=$2
onnx_path=$(get_real_path $1)
export CONFIG_PATH=${CONFIG_PATH}

if [ ! -f ${onnx_path} ]; then
    echo "Onnx file does not exist."
exit 1
fi

cd ../
CUDA_VISIBLE_DEVICES=0 python eval_onnx.py --onnx_path ${onnx_path} --config_path ${CONFIG_PATH}> eval.log 2>&1 &
