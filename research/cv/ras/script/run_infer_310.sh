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

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash run_infer_310.sh [GEN_MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID] [GT_PATH]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    DEVICE_ID is required and means the id of the device.
    GT_PATH is optional, it can be used to evaluate the results."
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

gen_model=$(get_real_path $1)
data_path=$(get_real_path $2)
preprocess_data_path="./preprocess_Data"

if [ "$3" == "y" ] || [ "$3" == "n" ];then
    need_preprocess=$3
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

if [ "$3" == "n" ]; then
    preprocess_data_path=$data_path
fi

device_id=$4

echo "generator mindir name: "$gen_model
echo "dataset path: "$data_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

need_eval=0
gt_path=""
if [ $# == 5 ]; then
    gt_path=$5
    need_eval=1
    echo "ground truth path: "$gt_path
fi

function preprocess_data()
{
    if [ -d $preprocess_data_path ]; then
        rm -rf $preprocess_data_path
    fi
    mkdir $preprocess_data_path
    mkdir $preprocess_data_path/data
    mkdir $preprocess_data_path/data_shape
    echo "Start to preprocess file..."
    python ../preprocess.py --dataroot=$data_path &> preprocess.log
    echo "file generates successfully!"
}

function compile_app()
{
    echo "Start to compile source code..."
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
    echo "Compile successfully."
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
    echo "Start to execute inference..."
    ../ascend310_infer/out/main --gen_mindir_path=$gen_model --dataset_path=$preprocess_data_path/ --device_id=$device_id &> infer.log
}

function postprocess_data()
{
    echo "Start to postprocess image file..."
    python ../postprocess.py --bin_path="./result_Files/" --shape_path=$preprocess_data_path/data_shape/ --target_path="./result_Files/" &> postprocess.log
    rm -rf ./result_Files/*.bin
}

function evaluation()
{
    echo "Start to evaluate..."
    python ../eval_310.py --pred_dir='./result_Files/' --gt_dir $gt_path &> evaluation.log
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess files failed"
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
    echo "execute inference failed"
    exit 1
fi

postprocess_data
if [ $? -ne 0 ]; then
    echo "postprocess images failed"
    exit 1
fi

if [ $need_eval -ne 0 ]; then
    evaluation
    if [ $? -ne 0 ]; then
        echo "evaluation images failed"
        exit 1
    fi
fi
