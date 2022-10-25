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

if [[ $# != 3 ]]; then
    echo "Usage:"
    echo "bash scripts/ascend310_inference.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]"
    exit 1
fi

get_real_path() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

model=$(get_real_path $1)
data_path=$(get_real_path $2)
device_id=$3

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "device id: "$device_id

function compile_app()
{
    cd ./ascend310_infer/src/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    bash build.sh &> build.log
}

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ./preprocess.py --data_path=$data_path --train_path=./preprocess_Result &> preprocess.log
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
    ./ascend310_infer/src/main --gmindir_path=$model --gdataset_path=./preprocess_Result/image --gdevice_id=$device_id  &> infer.log
}

function cal_acc()
{
    python ./postprocess.py --result_path=./result_Files --label_path=./preprocess_Result/label &> acc.log
    if [ $? -ne 0 ]; then
        echo "Calculate accuracy failed."
        exit 1
    fi
}

preprocess_data
if [ $? -ne 0 ]; then
    echo "Dataset preprocessing failed."
    exit 1
fi

compile_app
if [ $? -ne 0 ]; then
    echo "Compile app code failed."
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo "Execute inference failed."
    exit 1
fi

cal_acc
if [ $? -ne 0 ]; then
    echo "Calculate mIoU failed."
    exit 1
fi
