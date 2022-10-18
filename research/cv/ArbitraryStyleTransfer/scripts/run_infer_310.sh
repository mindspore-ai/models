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
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [CONTENT_PATH] [STYLE_PATH] [DEVICE_ID]
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
content_path=$(get_real_path $2)
style_path=$(get_real_path $3)
device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi
echo "mindir name: "$model
echo "content path: "$content_path
echo "style path: "$style_path
echo "device id: "$device_id

function preprocess_data()
{
   if [ -d preprocess_Result ]; then
       rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --content_path $content_path --style_path $style_path --output_path='./preprocess_Result/'
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
    if [ -d postprocess_Result_interpolation ]; then
       rm -rf ./postprocess_Result_interpolation
    fi
    mkdir postprocess_Result_interpolation
    python ../postprocess.py --bin_path='./result_Files' --output_dir='./postprocess_Result/' --output_dir_interpolation='./postprocess_Result_interpolation/' &> postprocess.log
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