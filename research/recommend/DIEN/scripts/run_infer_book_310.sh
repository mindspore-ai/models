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
    echo "Usage: bash run_infer_book_310.sh [MINDIR_PATH] [INPUT_PATH] [DEVICE_ID]"
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
input_path=$(get_real_path $2)

device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi

echo "mindir path: "$model
echo "input path: "$input_path
echo "device id: "$device_id

function compile_app()
{
    cd ../ascend310_infer
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
    ../ascend310_infer/out/main --mindir_path=$model --input0_path=$input_path/mid_mask --input1_path=$input_path/uids --input2_path=$input_path/mids --input3_path=$input_path/cats --input4_path=$input_path/mid_his --input5_path=$input_path/cat_his --input6_path=$input_path/noclk_mids --input7_path=$input_path/noclk_cats --device_id=$device_id &> infer_book.log
}

function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files --target_path=$input_path/target --dataset_type=Books &> acc_book.log &
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo "run inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
