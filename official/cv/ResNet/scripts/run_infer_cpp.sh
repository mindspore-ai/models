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

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "Usage: bash run_infer_cpp.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [CONFIG_PATH] [DEVICE_TYPE] [DEVICE_ID]
    NET_TYPE can choose from [resnet18, resnet34, se-resnet50, resnet50, resnet101, resnet152]
    DATASET can choose from [cifar10, imagenet]
    DEVICE_TYPE can choose from [Ascend, GPU, CPU]
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
if [ $2 == 'resnet18' ] || [ $2 == 'resnet34' ] || [ $2 == 'se-resnet50' ] || [ $2 == 'resnet50' ] || [ $2 == 'resnet152' ] || [ $2 == 'resnet101' ]; then
  network=$2
else
  echo "NET_TYPE can choose from [resnet18, se-resnet50]"
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

device_id=0
if [ $# == 7 ]; then
    device_id=$7
fi

if [ $6 == 'GPU' ]; then
    device_id=0
fi

# shellcheck disable=SC2153
if [ $6 == 'Ascend' ] || [ $6 == 'GPU' ] || [ $6 == 'CPU' ]; then
  device_type=$6
else
  echo "DEVICE_TYPE can choose from [Ascend, GPU, CPU]"
  exit 1
fi
echo "device type: "$device_type

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "network: "$network
echo "dataset: "$dataset
echo "device id: "$device_id

if [ $MS_LITE_HOME ]; then
  RUNTIME_HOME=$MS_LITE_HOME/runtime
  TOOLS_HOME=$MS_LITE_HOME/tools
  RUNTIME_LIBS=$RUNTIME_HOME/lib:$RUNTIME_HOME/third_party/glog/:$RUNTIME_HOME/third_party/libjpeg-turbo/lib
  RUNTIME_LIBS=$RUNTIME_LIBS:$RUNTIME_HOME/third_party/dnnl/
  export LD_LIBRARY_PATH=$RUNTIME_LIBS:$TOOLS_HOME/converter/lib:$LD_LIBRARY_PATH
  echo "Insert LD_LIBRARY_PATH the MindSpore Lite runtime libs path: $RUNTIME_LIBS $TOOLS_HOME/converter/lib"
fi


function compile_app()
{
    cd ../cpp_infer/src/ || exit
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
    python ../preprocess.py --data_path=$data_path --output_dir=./preprocess_Result --config_path=$config_path &> preprocess.log
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
    ../cpp_infer/src/main --device_type=$device_type --mindir_path=$model --dataset_path=$data_path --network=$network --dataset=$dataset --device_id=$device_id  &> infer.log
}

function cal_acc()
{
    if [ "x${dataset}" == "xcifar10" ] || [ "x${dataset}" == "xCifar10" ]; then
        python ../postprocess.py --dataset=$dataset --label_path=./preprocess_Result/label --result_path=result_Files --config_path=$config_path &> acc.log
    else
        python ../create_imagenet2012_label.py  --img_path=$data_path
        python ../postprocess.py --dataset=$dataset --result_path=./result_Files --label_path=./imagenet_label.json --config_path=$config_path &> acc.log
    fi
    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}

if [ "x${dataset}" == "xcifar10" ] || [ "x${dataset}" == "xCifar10" ]; then
    if [ $2 == 'resnet18' ]; then
        CONFIG_PATH=resnet18_cifar10_config.yaml
    else
        CONFIG_PATH=resnet50_cifar10_config.yaml
    fi
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
