#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_310.sh [SRCNN_MINDIR_PATH] [DATA_PATH] [OUTPUTS_PATH] [DEVICE_ID](optional) 
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

srcnn_model=$(get_real_path $1)
dataset_path=$(get_real_path $2)
outputs_path=$(get_real_path $3)

output_tensors=../ascend310_infer/data/output_tensors
input_tensors=../ascend310_infer/data/input_tensors

device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "generator mindir name: "$srcnn_model
echo "dataset path: "$dataset_path
echo "outputs path: "$outputs_path
echo "device id: "$device_id

function preprocess_data()
{
    echo "Start to preprocess..."
    if [ -d ../ascend310_infer/data ]; then
        rm -rf ../ascend310_infer/data
    fi
    mkdir ../ascend310_infer/data
    mkdir $input_tensors
    python ../preprocess.py --image_path=$dataset_path --output_path $input_tensors &> preprocess.log
    mv preprocess.log $logs_path
    echo "Preprocess successfully!"
}

function compile_app()
{
    echo "Start to compile source code..."
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
    mv build.log $logs_path
    echo "Compile successfully."
}

function infer()
{
    cd - || exit
    if [ -d $output_tensors ]; then
        rm -rf $output_tensors
    fi
    mkdir $output_tensors
    echo "Start to execute inference..."
    ../ascend310_infer/out/main --srcnn_mindir_path=$srcnn_model --dataset_path=$input_tensors --device_id=$device_id &> infer.log
    mv infer.log $logs_path
}

function postprocess_data()
{
    if [ -d $outputs_path ]; then
        rm -rf $outputs_path
    fi
    mkdir $outputs_path
    echo "Start to postprocess image file..."
    python ../postprocess.py --image_path=$dataset_path --predict_path=$output_tensors --result_path $outputs_path &> postprocess.log
    mv postprocess.log $logs_path
}

if [ -d logs ]; then
    rm -rf logs
fi
mkdir logs
logs_path=`pwd`/logs

preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess images failed"
    exit 1
fi

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo "execute inference failed"
    exit 1
fi

postprocess_data
if [ $? -ne 0 ]; then
    echo "postprocess images failed"
    exit 1
fi