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

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_TYPE] [DEVICE_ID]
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

dataset_path=$(get_real_path $2)
dataset_name="imagenet2012"
DVPP="CPU"

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
if [ $# -ge 1 ]; then
  if [ $dataset_name == 'cifar10' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  elif [ $dataset_name == 'imagenet2012' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_imagenet.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
else
  CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
fi


device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

if [ $3 == 'GPU' ]; then
    device_id=0
fi

echo "mindir name: "$model
echo "dataset path: "$dataset_path
echo "device id: "$device_id

function preprocess_data()
{
   if [ -d preprocess_Result ]; then
       rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --config_path=$CONFIG_FILE --dataset=$dataset_name --data_path=$dataset_path --result_path=./preprocess_Result/
}

if [ $3 == 'Ascend' ] || [ $3 == 'GPU' ] || [ $3 == 'CPU' ]; then
  device_type=$3
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
    cd ../cpp_infer/ || exit
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

    if [ "$DVPP" == "DVPP" ]; then
        ../cpp_infer/out/main --device_type=$device_type --mindir_path=$model --dataset_name=$dataset_name --input0_path=$dataset_path --device_id=$device_id --cpu_dvpp=../cpp_infer/aipp.cfg --image_height=256 --image_width=256  &> infer.log
    else
        ../cpp_infer/out/main --device_type=$device_type --mindir_path=$model --dataset_name=$dataset_name --input0_path=$dataset_path --device_id=$device_id &> infer.log
    fi
}

function cal_acc()
{

    python ../postprocess.py --config_path=$CONFIG_FILE --result_dir=./result_Files --label_dir=./preprocess_Result/imagenet_label.json   &> acc.log

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
