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
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [CONFIG_FILE] [DEVICE_ID]
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
model=$(get_real_path $1)
config_path=$(get_real_path $2)
device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi
echo "mindir name: "$model
echo "device id: "$device_id
echo "config_path: "$config_path

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
       rm -rf ./preprocess_Result
    fi

    if [ -d preprocess_Mask_Result ]; then
       rm -rf ./preprocess_Mask_Result
    fi
    mkdir preprocess_Result
    mkdir preprocess_Mask_Result
    python ../preprocess.py --config_path=$config_path &> preprocess.log
}
function compile_app()
{
    cd ../ascend310_infer/ || exit
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

    ../ascend310_infer/out/main --mindir_path=$model --input_path=./preprocess_Result --device_id=$device_id &> infer.log
}

function post_process()
{
   if [ -d postprocess_Result ]; then
       rm -rf ./postprocess_Result
    fi
    mkdir postprocess_Result
    python ../postprocess.py --bin_path='./result_Files/' --mask_path='./preprocess_Mask_Result/' --output_dir='./postprocess_Result/' &> postprocess.log
}

preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess dataset failed"
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

post_process
if [ $? -ne 0 ]; then
    echo " execute post_process failed"
    exit 1
fi
