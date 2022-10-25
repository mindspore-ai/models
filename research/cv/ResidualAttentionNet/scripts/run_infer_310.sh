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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [CONFIG_PATH] [DEVICE_ID]
    DATASET can choose from [cifar10, imagenet]
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

if [ $2 == 'cifar10' ] || [ $2 == 'imagenet' ]; then
  dataset=$2
else
  echo "DATASET can choose from [cifar10, imagenet]"
  exit 1
fi

data_path=$(get_real_path $3)
config_path=$(get_real_path $4)

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "dataset: "$dataset
echo "device id: "$device_id

function compile_app()
{
    cd ../ascend310_infer/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log
}

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python3.7 ../preprocess.py --data_path=$data_path --output_path=./preprocess_Result --config_path=$config_path &> preprocess.log
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
    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=$data_path --dataset=$dataset --device_id=$device_id  &> infer.log
}

function cal_acc()
{
    if [ "x${dataset}" == "xcifar10" ] || [ "x${dataset}" == "xCifar10" ]; then
        python ../postprocess.py --dataset=$dataset --label_path=./preprocess_Result/label --result_path=result_Files --config_path=$config_path &> acc.log
    else
        python3.7 ../create_imagenet2012_label.py  --img_path=$data_path
        python3.7 ../postprocess.py --dataset=$dataset --label_path=./imagenet_label.json --result_path=./result_Files  --config_path=$config_path &> acc.log
    fi
    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}

if [ "x${dataset}" == "xcifar10" ] || [ "x${dataset}" == "xCifar10" ]; then
    CONFIG_PATH=resnet50_cifar10_config.yaml
    preprocess_data ${CONFIG_PATH}
    data_path=./preprocess_Result/img_data
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