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
if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [TEST_LR_PATH] [TEST_GT_PATH] [NEED_PREPROCESS] [DEVICE_ID]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
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
test_LR_path=$(get_real_path $2)
test_GT_path=$(get_real_path $3)

if [ "$4" == "y" ] || [ "$4" == "n" ];then
    need_preprocess=$4
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

echo "mindir name: "$model
echo "test_LR_path: "$test_LR_path
echo "test_GT_path: "$test_GT_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d preprocess_path ]; then
        rm -rf ./preprocess_path
    fi
    mkdir preprocess_path
    python ../preprocess.py --test_LR_path=$test_LR_path --test_GT_path=$test_GT_path  --result_path=./preprocess_path/
}

function compile_app()
{
    cd ../ascend310_infer || exit
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

    ../ascend310_infer/out/main --mindir_path=$model --input0_path=./preprocess_path --device_id=$device_id &> infer.log

}

function cal_acc()
{
    if [ -d infer_output ]; then
        rm -rf ./infer_output
    fi
    mkdir infer_output
    python ../postprocess.py --test_LR_path=$test_LR_path --test_GT_path=$test_GT_path --device_id=$device_id &> acc.log
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
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
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi