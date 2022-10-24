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

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
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
dataset_name=$2
dataset_path=$(get_real_path $3)
device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "mindir name: "$model
echo "dataset name: "$dataset_name
echo "dataset path: "$dataset_path
echo "device id: "$device_id

function preprocess_data()
{
    python ../convert_data.py --data_url $dataset_path --output_file ./new_data.tmp
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

    ../ascend310_infer/out/sdne $model $dataset_name new_data.tmp $device_id  &> infer.log
    rm -rf new_data.tmp
}

function cal_acc()
{
    python ../postprocess.py --data_path $dataset_path --dataset $dataset_name --emb_file ../scripts/result_Files/embeddings.txt --rec_file ../scripts/result_Files/reconstructions.txt &> acc.log
}

preprocess_data
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
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
