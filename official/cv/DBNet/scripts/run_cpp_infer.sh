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
    echo "Usage: bash run_cpp_infer.sh [MINDIR_PATH] [CONFIG_PATH] [OUTPUT_DIR] [DEVICE_TARGET] [DEVICE_ID]
    DEVICE_TARGET can choose from [Ascend, GPU, CPU]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
MINDIR_PATH=$(get_real_path $1)
CONFIG_PATH=$(get_real_path $2)
OUTPUT_DIR=$(get_real_path $3)
DEVICE_TARGET=$4
DEVICE_ID=0
if [ $# == 5 ]; then
    DEVICE_ID=$5
fi

echo "mindir name: "$MINDIR_PATH
echo "config path: "$CONFIG_PATH
echo "output path: "$OUTPUT_DIR
echo "device target: "$DEVICE_TARGET
echo "device id: "$DEVICE_ID


if [ $MS_LITE_HOME ]; then
  RUNTIME_HOME=$MS_LITE_HOME/runtime
  TOOLS_HOME=$MS_LITE_HOME/tools
  RUNTIME_LIBS=$RUNTIME_HOME/lib:$RUNTIME_HOME/third_party/glog/:$RUNTIME_HOME/third_party/libjpeg-turbo/lib
  export LD_LIBRARY_PATH=$RUNTIME_LIBS:$TOOLS_HOME/converter/lib:$LD_LIBRARY_PATH
  echo "Insert LD_LIBRARY_PATH the MindSpore Lite runtime libs path: $RUNTIME_LIBS $TOOLS_HOME/converter/lib"
fi

function preprocess_data()
{
    output_dir=$OUTPUT_DIR
    if [ -d $output_dir ]; then
        rm -rf $output_dir
    fi
    mkdir $output_dir
    python $BASE_PATH/../src/utils/gen_cpp_infer_inputs.py \
        --output_dir=$output_dir \
        --config_path=$CONFIG_PATH &> $output_dir/preprocess.log
}

function compile_app()
{
    cd $BASE_PATH/../cpp_infer/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    bash build.sh &> $OUTPUT_DIR/build.log
    cd -
}

function infer()
{
    cd $OUTPUT_DIR || exit
    if [ -d eval_result_bin ]; then
        rm -rf ./eval_result_bin
    fi
    if [ -d time_result ]; then
        rm -rf ./time_result
    fi
    mkdir eval_result_bin
    mkdir time_result
    $BASE_PATH/../cpp_infer/main \
        --mindir_path=$MINDIR_PATH \
        --dataset_path=$OUTPUT_DIR/eval_input_bin \
        --output_path=$OUTPUT_DIR/eval_result_bin \
        --device_type=$DEVICE_TARGET \
        --device_id=$DEVICE_ID  &> $OUTPUT_DIR/infer.log
    cd -
}

function cal_acc()
{
    python $BASE_PATH/../src/utils/gen_cpp_infer_result.py \
        --output_dir=$OUTPUT_DIR \
        --config_path=$CONFIG_PATH &> $OUTPUT_DIR/result.log
}

preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess data code failed"
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
