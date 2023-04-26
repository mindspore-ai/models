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
  echo "Usage: bash run_infer_cpp.sh [MODEL_PATH] [DATA_PATH] [CLS_NAME] [DEVICE_TYPE] [DEVICE_ID]
  DEVICE_TYPE can choose from [Ascend, GPU, CPU]
  DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
  exit 1
fi

if [ ! -f $1 ]
then
  echo "error: MODEL_PATH=$1 is not a file"
  exit 1
fi

if [ ! -d $2 ]
then
  echo "error: DATA_PATH=$2 is not a folder"
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
cls_name=$3

if [ $# == 5 ]; then
    device_id=$5
elif [ $# == 4 ]; then
    if [ -z $device_id ]; then
        device_id=0
    else
        device_id=$device_id
    fi
fi

if [ $4 == 'GPU' ]; then
    if [ $CUDA_VISIABLE_DEVICES ]; then
        device_id=$CUDA_VISIABLE_DEVICES
    fi
fi

echo model: $model
echo data_path: $data_path
echo cls_name: $cls_name
echo device_id: $device_id

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
    pushd cpp_infer
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    popd
}

function infer()
{
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    mkdir -p time_Result
    mkdir result_Files
    cpp_infer/out/main --device_type=$device_type --model_path=$model --dataset_path=$data_path --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

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

echo "Compile Ransac Voting"

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
echo ${BASEPATH}
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

ransac_voting_path="${BASEPATH}/../src/lib/voting"
pushd ${ransac_voting_path}
echo ${ransac_voting_path}
bash run_setup.sh
echo "Compiling Finished"
popd
result_path="${BASEPATH}/../result_Files/"
echo "result_Files dir: "${result_path}
python postprocess.py --result_path=$result_path --cls_name=$cls_name &> postprocess.log