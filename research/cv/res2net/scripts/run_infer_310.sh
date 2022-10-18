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

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [CONFIG_PATH] [NEED_PREPROCESS] [DEVICE_ID]
    NET_TYPE can choose from [res2net50, res2net152, res2net101, se-res2net50]
    DATASET can choose from [cifar10, imagenet]
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
if [ $2 == 'res2net50' ] || [ $2 == 'res2net152' ] || [ $2 == 'res2net101' ] || [ $2 == 'se-res2net50' ] ; then
  network=$2
else
  echo "NET_TYPE can choose from [res2net50, res2net152, res2net101, se-res2net50]"
  exit 1
fi

if [ $3 == 'cifar10' ] || [ $3 == 'imagenet' ]; then
  dataset=$3
else
  echo "DATASET can choose from [cifar10, imagenet]"
  exit 1
fi

data_path=$(get_real_path $4)
config_path=$(get_real_path $5)

if [ "$6" == "y" ] || [ "$6" == "n" ];then
    need_preprocess=$6
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 7 ]; then
    device_id=$7
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "network: "$network
echo "config path: "$config_path
echo "dataset: "$dataset
echo "need preprocess:"$need_preprocess
echo "device id: "$device_id

function compile_app()
{
    cd ../ascend310_infer/ || exit
    bash build.sh &> build.log
}

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --data_path=$data_path --output_path=./preprocess_Result --config_path=$config_path --dataset $dataset &> preprocess.log
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
    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=$data_path --network=$network --dataset=$dataset --device_id=$device_id  &> infer.log
}

function cal_acc()
{
    if [ "${dataset}" == "cifar10" ] ; then
        python ../postprocess.py --dataset=$dataset --label_path=./preprocess_Result/label.npy --result_path=result_Files --config_path=$config_path &> acc.log
    else        
        python ../postprocess.py --dataset=$dataset --result_path=./result_Files --label_path=./preprocess_Result/imagenet_label.json --config_path=$config_path --batch_size=1 &> acc.log
    fi
    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}


if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ "${dataset}" == "cifar10" ] ; then    
        data_path=./preprocess_Result/img_data
    fi
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
