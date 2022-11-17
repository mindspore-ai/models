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


run()
{
    # check pretrain_ckpt file
    if [ ! -f $3 ]
    then
        echo "error: ONNX_MODEL_PATH=$3 is not a file"
    exit 1
    fi

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    CONFIG_FILE="${BASEPATH}/../$4"
    export PYTHONPATH=${BASEPATH}:$PYTHONPATH

    python ${BASEPATH}/../eval_onnx.py \
        --config_path=$CONFIG_FILE \
        --platform=$1 \
        --dataset_path=$2 \
        --file_name=$3 \
        &> ../eval_onnx.log &
}

if [ $# -ne 3 ]
then
    echo "Usage:
          bash run_eval_onnx.sh [PLATFORM] [DATASET_PATH] [ONNX_MODEL_PATH]"
exit 1
fi

# check dataset path
if [ ! -d $2 ]
then
    echo "error: DATASET_PATH=$2 is not a directory"
    exit 1
fi

if [ $1 = "CPU" ] ; then
    CONFIG_FILE="default_config_cpu.yaml"
elif [ $1 = "GPU" ] ; then
    CONFIG_FILE="default_config_gpu.yaml"
else
    echo "Unsupported platform."
fi;

run "$@" $CONFIG_FILE
