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

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash run_infer_cpp.sh [NETWORK] [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID]
    DEVICE_TYPE can choose from [Ascend, GPU, CPU]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero.
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'."
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
if [ "$3" == "y" ] || [ "$3" == "n" ];then
    need_preprocess=$3
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

if [ $4 == 'GPU' ]; then
    device_id=0
fi

echo "network: " $network
echo "mindir name: "$model
echo "need preprocess or not: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    if [ $network == "unet" ]; then
        config_path="${BASEPATH}/../unet_simple_config.yaml"
    elif [ $network == "unet++" ]; then
        config_path="${BASEPATH}/../unet_nested_cell_config.yaml"
    else
        echo "unsupported network"
        exit 1
    fi
    python ../preprocess.py --config_path=$config_path
}

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


function compile_app()
{
    cd ../cpp_infer/src || exit
    if [ -f "Makefile" ]; then
        make clean
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
    ../cpp_infer/src/main --device_type=$device_type --mindir_path=$model --dataset_path=./preprocess_Result/ --device_id=$device_id --need_preprocess=$need_preprocess &> infer.log
}

function cal_acc()
{
    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    if [ $network == "unet" ]; then
        config_path="${BASEPATH}/../unet_simple_config.yaml"
    elif [ $network == "unet++" ]; then
        config_path="${BASEPATH}/../unet_nested_cell_config.yaml"
    fi
    python ../postprocess.py --config_path=$config_path &> acc.log &
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
    echo "execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
