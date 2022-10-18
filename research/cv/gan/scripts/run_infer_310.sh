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
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ -z "$1" ]; then
        echo "get empty path"
    elif [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
input_path=$(get_real_path $2)
device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi

echo "mindir name: "$model
echo "input path: "$input_path
echo "device id: "$device_id

function preprocess_data()
{
    cd ..
    python ./preprocess.py
    cd - || exit
}

function compile_app()
{
    cd ../ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log
    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    echo "compile app finish"
    cd - || exit
}


function infer()
{
    cd ../ascend310_infer || exit
    if [ -d result_files ]; then
        rm -rf ./result_files
    fi
    mkdir result_files
    ./out/gan --mindir_path=$model --input0_path=./data_dir --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
    echo "execute inference success"
}

function cal_acc()
{
    cd ..
    if [ -d images ]; then
        rm -rf ./images
    fi
    mkdir images
    nohup python -u postprocess.py --data_path=$input_path > acc.log 2>&1 &
    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
    echo "postprocess success"
}
preprocess_data
compile_app
infer
cal_acc
