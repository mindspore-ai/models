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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET] [INPUT_PATH] [DEVICE_ID]"
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
dataset=$2
input_path=$(get_real_path $3)
device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "mindir name: "$model
echo "dataset: "$dataset
echo "input_path: "$input_path
echo "device id: "$device_id

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function preprocess()
{
    if [ -d output ]; then
        rm -rf ./output
    fi
    mkdir output
    python ../preprocess.py --dataset=$dataset --test_dir $input_path
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
    ../ascend310_infer/out/main --mindir_path=$model --input0_path=./output --device_id=$device_id  &> infer.log
}
function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files --dataset=$dataset --test_data_path=$input_path &> acc.log &
}

preprocess
if [ $? -ne 0 ]; then
    echo "preprocess data failed"
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
cal_acc
wait
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
echo "finished "