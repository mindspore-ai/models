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

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: bash run_infer_cpp.sh [MINDIR_PATH] [CONFIG_PATH] [SSIM_THRESHOLD] [L1_THRESHOLD] [DEVICE_TYPE] [DEVICE_ID]
    DEVICE_TYPE can choose from [Ascend, GPU, CPU]
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

model=$(get_real_path $1)
config_path=$(get_real_path $2)
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
need_preprocess="n"
ssim_threshold=$3
l1_threshold=$4

device_id=0
if [ $# == 6 ]; then
    device_id=$6
fi

if [ $5 == 'GPU' ]; then
    device_id=0
fi

echo "mindir name: "$model
echo "config path: "$config_path
echo "device id: "$device_id
echo "ssim threshold: "$ssim_threshold
echo "l1 threshold: "$l1_threshold

function preprocess_data()
{
    cd $BASE_PATH/.. || exit
    if [ -d ./preprocess_result ]; then
        rm -rf ./preprocess_result
    fi
    python $BASE_PATH/../preprocess.py --config_path=$config_path  &> pre.log
}


if [ $5 == 'Ascend' ] || [ $5 == 'GPU' ] || [ $5 == 'CPU' ]; then
  device_type=$5
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


function compile_app()
{
    cd $BASE_PATH/../cpp_infer || exit
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
    cd $BASE_PATH/.. || exit
    if [ -d ./postprocess_result ]; then
        rm -rf ./postprocess_result
    fi
    mkdir ./postprocess_result
    ./cpp_infer/out/main --device_type=$device_type --mindir_path=$model --dataset_path=./preprocess_result --device_id=$device_id --need_preprocess=$need_preprocess &> infer.log
}

function cal_acc()
{
    cd $BASE_PATH/.. || exit
    if [ -d save_img ]; then
        rm -rf ./save_img
    fi
    mkdir save_img
    python postprocess.py --config_path=$config_path --ssim_threshold=$ssim_threshold --l1_threshold=$l1_threshold --save_dir=save_img &> acc.log
}

preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess dataset failed"
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