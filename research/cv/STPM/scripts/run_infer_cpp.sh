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
if [[ $# != 6 ]]; then
    echo "Usage: bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID] [CATEGORY] [DEVICE_TYPE]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    DEVICE_TYPE can choose from [Ascend, GPU, CPU]"
exit 1
fi

get_real_path() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

model=$(get_real_path $1)
dataset_path=$(get_real_path $2)

if [ "$3" == "y" ] || [ "$3" == "n" ]; then
    need_preprocess=$3
else
    echo "weather need preprocess or not, it's value must be in [y, n]"
    exit 1
fi

device_id=$4
category=$5

if [ $6 == 'GPU' ]; then
    device_id=0
fi

echo "Mindir name: "$model
echo "dataset path: "$dataset_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id
echo "category: "$category

function preprocess_data() {
    if [ ! -d img ]; then
        mkdir ./img
    fi
    if [ -d img/$category ]; then
        rm -rf img/$category
    fi
    mkdir ./img/$category
    mkdir ./img/$category/label

    python ../preprocess.py \
    --data_dir $dataset_path \
    --img_dir ./img/$category \
    --category $category
}

if [ $6 == 'Ascend' ] || [ $6 == 'GPU' ] || [ $6 == 'CPU' ]; then
  device_type=$6
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


function compile_app() {
    cd ../cpp_infer || exit
    bash build.sh &> build.log
}

function infer() {
    cd - || exit
    if [ -d img/$category/result ]; then
        rm -rf img/$category/result
    fi
    mkdir img/$category/result

    if [ -d img/$category/time ]; then
        rm -rf img/$category/time
    fi
    mkdir img/$category/time

    ../cpp_infer/out/main \
        --device_type=$device_type --mindir_path=$model \
        --input_path=./img/$category \
        --result_path=./img/$category/result \
        --time_path=./img/$category/time \
        --device_id=$device_id &> infer_$category.log
}

function cal_acc() {
    python ../postprocess.py \
        --result_dir ./img/$category/result/ \
        --data_dir $dataset_path \
        --label_dir ./img/$category/label/ \
        --category $category > acc_$category.log
}

if [ $need_preprocess == "y" ]; then
   preprocess_data
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
