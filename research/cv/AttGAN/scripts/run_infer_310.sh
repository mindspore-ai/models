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
    echo "Usage: bash run_infer_310.sh [GEN_MINDIR_PATH] [ATTR_FILE_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
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

gen_model=$(get_real_path $1)
attr_path=$(get_real_path $2)
data_path=$(get_real_path $3)

if [ "$4" == "y" ] || [ "$4" == "n" ];then
    need_preprocess=$4
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

echo "generator mindir name: "$gen_model
echo "attribute file path: "$attr_path
echo "dataset path: "$data_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    echo "Start to preprocess attr file..."
    python ../preprocess.py --attrs_path=$attr_path --test_int=1.0 --thres_int=0.5 &> preprocess.log
    echo "Attribute file generates successfully!"
}

function compile_app()
{
    echo "Start to compile source code..."
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
    echo "Compile successfully."
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
    echo "Start to execute inference..."
    ../ascend310_infer/out/main --gen_mindir_path=$gen_model --dataset_path=$data_path --attr_file_path="attrs.txt" --device_id=$device_id --image_height=128 --image_width=128 &> infer.log
}

function postprocess_data()
{
    echo "Start to postprocess image file..."
    python ../postprocess.py --bin_path="./result_Files/" --target_path="./result_Files/"
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess attrs failed"
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
    echo "execute inference failed"
    exit 1
fi

postprocess_data
if [ $? -ne 0 ]; then
    echo "postprocess images failed"
    exit 1
fi
