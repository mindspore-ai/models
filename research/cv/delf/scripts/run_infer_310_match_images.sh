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


if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_infer_310.sh [GEN_MINDIR_PATH] [IMAGES_PATH] [DEVICE_ID]
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
data_path=$(get_real_path $2)
image_list="./list_images.txt"

device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi

echo "generator mindir name: "$gen_model
echo "dataset path: "$data_path
echo "dataset path: "$image_list
echo "device id: "$device_id

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi



function preprocess_data()
{
    echo "Start to preprocess images..."
    rm -rf ./preprocess_images
    rm -rf ./image_pyramids
    python ./src/preprocess.py --use_list_txt="True" --list_images_path=$image_list \
    --images_path=$data_path --output_path="./preprocess_images" --size_path="./image_pyramids" &> preprocess.log
    echo "Images generates successfully!"
}


function compile_app()
{
    echo "Start to compile source code..."
    cd ./ascend310_infer || exit
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
    ./ascend310_infer/out/main --gen_mindir_path=$gen_model --dataset_path="./preprocess_images" \
    --device_id=$device_id &> infer.log
}

function postprocess_data()
{
    rm -rf ./feature_Files
    echo "Start to postprocess image file..."
    python ./src/postprocess.py --use_list_txt="True" --list_images_path=$image_list \
    --bin_path="./result_Files/" --target_path="./feature_Files/" --images_path=$data_path \
    --size_path="./image_pyramids" &> postprocess.log
}

function match_images()
{
    echo "Start to match images..."
    python ./src/match_images.py --list_images_path=$image_list \
    --images_path=$data_path --feature_path="./feature_Files/" \
    --output_image="./test_match.png" &> match_images.log
}

preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess failed"
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

postprocess_data
if [ $? -ne 0 ]; then
    echo "postprocess images failed"
    exit 1
fi

match_images
if [ $? -ne 0 ]; then
    echo "match images failed"
    exit 1
fi
