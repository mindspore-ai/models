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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [DEVICE_ID]
    DEVICE_ID is optional, default value is zero,
    [DATA_PATH] is the path of dataset,
    [DATASET] is [market1501|dukemtmcreid|cuhk03|msmt17]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

typeset -l dataset
model=$(get_real_path $1)
dataset=$2
dataset_dir=$(get_real_path $3)
device_id=0

if [ $# == 4 ]; then
    device_id=$4
fi

echo $model
echo $dataset
echo $device_id

function compile_app()
{
    cd ../ascend310_infer || exit
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

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    config_path="${BASEPATH}/../osnet_config.yaml"
    python ../preprocess.py --config_path=$config_path --data_path=$dataset_dir --batch_size_test=1 --target=$dataset --output_path=./preprocess_Result
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
    ../ascend310_infer/out/main --model_path=$model --dataset=$dataset --dataset_path=$data_path --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function cal_acc()
{

    python ../postprocess.py --dataset=$dataset --label_file=./preprocess_Result/label --camlabel_file=./preprocess_Result/camlabel \
    --result_path=result_Files &> acc.log

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
