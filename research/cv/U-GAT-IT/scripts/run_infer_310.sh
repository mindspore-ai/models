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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
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

test_path=$2

device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi

echo "mindir name: "$model
echo "test data: "$test_path
echo "device id: "$device_id

function compile_app()
{
    cd ascend310_infer || exit
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

    ascend310_infer/out/main --mindir_path=$model --input0_path=$test_path --device_id=$device_id --fusion_switch_path=./ascend310_infer/fusion_switch.cfg &> infer.log

}

function generate_img()
{
    if [ -d infer_output_img ]; then
        rm -rf ./infer_output_img
    fi
    mkdir infer_output_img
    python postprocess.py  --bifile_outputdir=./result_Files --eval_outputdir=./infer_output_img --device_id=$device_id &> acc.log
}

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
generate_img
if [ $? -ne 0 ]; then
    echo "generate images failed"
    exit 1
fi
