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
    echo "Usage: bash run_infer_310_2d.sh [NETWORK] [MINDIR_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero."
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

network=$1
model=$(get_real_path $2)


device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi

echo "network: " $network
echo "mindir name: "$model
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d ./preprocess_Result ]; then
         echo "File location is right, continue next"
    else
         echo "Please refer to readme for data organization"
         return 1
    fi

    if [ $network == "nnUNet_2d" ]; then
        echo "2d network inference"
    elif [ $network == "nnUNet_3d_fullres" ]; then
        echo "3d fullres network inference"
    else
        echo "unsupported network"
        exit 1
    fi

}

function compile_app()
{
    cd ../ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    if [ -d out ]; then
        rm -rf ./out
    fi
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
    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=../preprocess_Result/bboxes --device_id=$device_id  &> infer.log
}

function cal_acc()
{   cd ../ || exit
    python postprocess.py network &> scripts/acc.log
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
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess_data failed"
    exit 1
fi

