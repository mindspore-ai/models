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
    echo "Usage: bash run_infer_cpp.sh [ENCODER_PATH] [DECODER_PATH] [DATA_PATH] [DEVICE_TYPE] [DEVICE_ID]
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

encoder=$(get_real_path $1)
decoder=$(get_real_path $2)
data_path=$(get_real_path $3)

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
    device_id=0
fi

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

if [ ! -f $encoder ]
then 
    echo "error: ENCODER_PATH=$encoder is not a file"
exit 1
fi

if [ ! -f $decoder ]
then 
    echo "error: DECODER_PATH=$decoder is not a file"
exit 1
fi

if [ ! -d $data_path ]
then 
    echo "error: DATA_PATH=$data_path is not a directory"
exit 1
fi

function compile_app()
{
    cd ../cpp_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "error: compile app code failed"
        exit 1
    fi
    cd - || exit
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
    ../cpp_infer/out/main --device_type=$device_type --encoder_path=$encoder --decoder_path=$decoder --dataset_path=$data_path --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "error: execute inference failed"
        exit 1
    fi
}

compile_app
infer
python ../cal_metrics.py $data_path result_Files >> infer.log  2>&1