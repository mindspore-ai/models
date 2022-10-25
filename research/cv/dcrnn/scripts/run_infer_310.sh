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

if [[ $# != 3 ]]; then
    echo "Usage:"
    echo "bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]"
    echo "DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
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
data_path=$(get_real_path $2)
device_id=$3

device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi


echo "mindir name: "$model
echo "dataset: "$data_path
echo "device id: "$device_id

function preprocess_data()
{
    cd ../ || exit
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python preprocess.py --init-dataset-path=$data_path &> pre_info.log
}

function compile_app()
{
    cd ./ascend_310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    bash build.sh &> build.log
}

function infer()
{
    cd ../scripts || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend_310_infer/out/dcrnn --mindir_path=$model --dataset_path=../preprocess_Result/data --device_id=$device_id --fusion_switch_path=../fusion_switch.cfg &> infer.log
}

function cal_acc()
{
    python ../postprocess.py --label-path=../preprocess_Result/label --data-path=./result_Files &> acc_info.log
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
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
