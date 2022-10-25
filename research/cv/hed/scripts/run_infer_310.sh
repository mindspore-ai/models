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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
    DEVICE_TARGET must choose from ['GPU', 'CPU', 'Ascend']
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

if [ "$2" == "y" ] || [ "$2" == "n" ];then
    need_preprocess=$2
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi


echo "mindir name: "$model
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    echo "preprocess_data doing..."
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py
    echo "preprocess_data done."
}

function compile_app()
{
    echo "compile_app doing..."
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
    echo "compile_app done."
}

function infer()
{
    echo "infer doing..."
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result

    ../ascend310_infer/out/main --mindir_path=$model --input0_path=./preprocess_Result/00_images --device_id=$device_id &> infer.log

    echo "infer done."
}

function cal_acc()
{
    echo "cal_acc doing"
    python ../postprocess.py &> acc.log
    echo "cal_acc done."
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess dataset failed"
        exit 1
    fi
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
