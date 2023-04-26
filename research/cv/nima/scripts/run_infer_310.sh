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
    echo "Usage: bash run_infer_310.sh [MODEL_PATH] [VAL_DATA_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: MODEL_PATH=$1 is not a file"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: DATA_PATH=$2 is not a folder"
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

if [ $# == 3 ]; then
    device_id=$3
elif [ $# == 2 ]; then
    if [ -z $device_id ]; then
        device_id=0
    else
        device_id=$device_id
    fi
fi

echo $model
echo $data_path
echo $device_id

function compile_app()
{
    cd ./ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    cd - || exit
}

function infer()
{
     if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir time_Result
    ./ascend310_infer/out/main --model_path=$model --dataset_path=$data_path --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

compile_app
infer
