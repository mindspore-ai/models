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
    MINDIR_PATH is mandatory, and must specify mindir path used
    DATA_PATH is mandatory, and must specify dataset path used
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
data_path=$(get_real_path $2)

device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi

echo "mindir name: "$model
echo "data path: "$data_path
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --data_path=$data_path --result_path=./preprocess_Result/
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

    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=./preprocess_Result/image --cpu_dvpp='CPU' --device_id=$device_id --image_height=96 --image_width=96 &> infer.log

}

function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files --data_path=./preprocess_Result/image --label_path=./preprocess_Result/label  &> acc.log
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
