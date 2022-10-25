#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http//www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 4 ]
then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [YAML_PATH] [DATA_PATH] [DEVICE_ID]"
    echo "Example: bash run_infer_310.sh ./PSPNet.mindir ./config/voc2012_pspnet50.yaml ./data/voc/ 0"
    exit 1
fi

MINDIR_PATH=$1
YAML_PATH=$2
DATA_PATH=$3
DEVICE_ID=$4

echo "MINDIR_PATH: "$MINDIR_PATH
echo "YAML_PATH: "$YAML_PATH
echo "DATA_PATH: "$DATA_PATH
echo "DEVICE_ID: "$DEVICE_ID

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python preprocess.py --config=$YAML_PATH --save_path=./preprocess_Result/ --data_path=$DATA_PATH &> preprocess.log
}

function compile_app()
{
    cd ./ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ./ascend310_infer/out/main --mindir_path=$MINDIR_PATH --input0_path=./preprocess_Result/inputs.txt --dims_save_path=./result_Files/ --device_id=$DEVICE_ID &> infer.log
}

function postprocess_data()
{
    if [ -d postprocess_Result ]; then
        rm -rf ./postprocess_Result
    fi
    mkdir postprocess_Result
    python postprocess.py --config=$YAML_PATH --data_path=$DATA_PATH &> postprocess.log
}

preprocess_data
if [ $? -ne 0 ]; then
    echo "prepocess data failed"
    exit 1
fi

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi

postprocess_data
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
