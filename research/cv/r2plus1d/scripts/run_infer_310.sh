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

if [ $# != 3 ]; then
    echo "Usage: bash run_infer_310.sh [model_path] [data_path] [out_image_path]"
exit 1
fi

get_real_path_name() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

get_real_path() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)/"
    fi
}

model=$(get_real_path_name $1)
data_path=$(get_real_path $2)
out_image_path=$(get_real_path $3)

echo "model path: "$model
echo "dataset path: "$data_path
echo "out image path: "$out_image_path

function compile_app()
{
    cd ../ascend310_infer/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    bash build.sh &> build.log
}

function preprocess()
{
    cd ../ || exit
    python preprocess.py --output_path=$out_image_path --dataset_root_path=$data_path &> preprocess.log
}

function infer()
{
    cd ./ascend310_infer || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ./out/main $model $out_image_path &> infer.log
}

function postprocess()
{
    python ../postprocess.py --result_path=./result_Files &> acc.log
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

preprocess
if [ $? -ne 0 ]; then
    echo "execute preprocess failed"
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi

postprocess
if [ $? -ne 0 ]; then
    echo "calculate acc failed"
    exit 1
fi
