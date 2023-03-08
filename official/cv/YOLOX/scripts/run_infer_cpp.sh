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
# ===========================================================================
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_cpp.sh [MINDIR_PATH] [DATA_DIR] [DEVICE_TYPE] [DEVICE_ID]
    DEVICE_TYPE can choose from [Ascend, GPU, CPU]
    DEVICE_ID is optional, default value is zero."
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

#typeset -l dataset
model=$(get_real_path $1)
data_dir=$(get_real_path $2)
device_id=0

if [ $# == 4 ]; then
    device_id=$4
fi

if [ $3 == 'GPU' ]; then
    device_id=0
fi

echo $model
echo $data_dir
echo $device_id

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
    cd ../cpp_infer || exit
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

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --output_path=./preprocess_Result --data_dir=$data_dir
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../cpp_infer/out/main --device_type=$device_type --model_path=$model --dataset_path=$data_path --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function cal_acc()
{
    python ../postprocess.py --result_path=result_Files --data_dir=$data_dir &> acc.log

    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}

preprocess_data
data_path=./preprocess_Result/img_data
compile_app
infer
cal_acc