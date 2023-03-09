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
    echo "Usage: bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [DEVICE_TYPE] [DEVICE_ID]
    DVPP is mandatory, and must choose from [DVPP|CPU|N], it's case-insensitive
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
data_path=$(get_real_path $2)
DVPP=${3^^}

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

if [ $4 == 'GPU' ]; then
    device_id=0
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "image process mode: "$DVPP
echo "device id: "$device_id

if [ $4 == 'Ascend' ] || [ $4 == 'GPU' ] || [ $4 == 'CPU' ]; then
  device_type=$4
else
  echo "DEVICE_TYPE can choose from [Ascend, GPU, CPU]"
  exit 1
fi
echo "device type: "$device_type

if [ $MS_LITE_HOME ]; then
  RUNTIME_HOME=$MS_LITE_HOME/runtime
  TOOLS_HOME=$MS_LITE_HOME/tools
  RUNTIME_LIBS=$RUNTIME_HOME/lib:$RUNTIME_HOME/third_party/glog/:$RUNTIME_HOME/third_party/libjpeg-turbo/lib
  RUNTIME_LIBS=$RUNTIME_LIBS:$RUNTIME_HOME/third_party/dnnl/
  export LD_LIBRARY_PATH=$RUNTIME_LIBS:$TOOLS_HOME/converter/lib:$LD_LIBRARY_PATH
  echo "Insert LD_LIBRARY_PATH the MindSpore Lite runtime libs path: $RUNTIME_LIBS $TOOLS_HOME/converter/lib"
fi

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --dataset_path=$data_path --output_path=./preprocess_Result &> preprocess.log
    data_path=./preprocess_Result/00_data
    label_path=./preprocess_Result/labels_ids.npy
}

function compile_app()
{
    cd ../cpp_infer || exit
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
    if [ "$DVPP" == "DVPP" ];then
      ../cpp_infer/out/main --device_type=$device_type --mindir_path=$model --dataset_path=$data_path --device_id=$device_id --cpu_dvpp=$DVPP --aipp_path=../src/aipp.cfg --image_height=32 --image_width=32 &> infer.log
    elif [ "$DVPP" == "CPU"  ]; then
      ../cpp_infer/out/main --device_type=$device_type --mindir_path=$model --dataset_path=$data_path --cpu_dvpp=$DVPP --device_id=$device_id --image_height=32 --image_width=32 &> infer.log
    elif [ "$DVPP" == "N" ];then
      ../cpp_infer/out/main --device_type=$device_type --mindir_path=$model --dataset_path=$data_path --device_id=$device_id --cpu_dvpp=$DVPP --image_height=32 --image_width=32 &> infer.log
    else
      echo "image process mode must be in [DVPP|CPU|N]"
      exit 1
    fi
}

function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files --label_path=$label_path &> acc.log &
}

# preprocess_data
preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess data failed"
    exit 1
fi
echo "reprocess data success"

echo "compiling app"
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
echo "successfully complied app"

echo "inferring"
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
echo "successfully inferred"

echo "calculating acc"
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi