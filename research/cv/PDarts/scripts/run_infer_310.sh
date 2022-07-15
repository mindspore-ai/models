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

if [ $# != 2 ]
then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH]"
exit 1
fi

export DEVICE_ID=0
export RANK_SIZE=1

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
dataset_name='cifar10'


dataset_path=$(get_real_path $2)

BASEPATH=$(dirname "$(pwd)") 
echo "base path :"$BASEPATH
echo "mindir name: "$model
echo "dataset name: "$dataset_name
echo "dataset path: "$dataset_path

export driver_home=/usr/local/Ascend
export install_path=${driver_home}/ascend-toolkit/latest
export DDK_PATH=${install_path}
export PATH=/usr/local/python3.7.5/bin:${install_path}/toolkit/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:${install_path}/acllib/lib64:${install_path}/atc/lib64:${install_path}/fwkacllib/lib64:${driver_home}/driver/lib64:${driver_home}/add-ons:${LD_LIBRARY_PATH}
export PYTHONPATH=${install_path}/pyACL/python/site-packages:${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:${install_path}/toolkit/python/site-packages:${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:${install_path}/opp/op_impl/built-in/ai_core/tbe:${install_path}/toolkit/latest/acllib/lib64:${PYTHONPATH}
export ASCEND_OPP_PATH=${install_path}/opp
export NPU_HOST_LIB=${install_path}/acllib/lib64/stub
export SOC_VERSION=Ascend310

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python preprocess.py --data_path=$dataset_path 
}

function compile_app()
{
    cd ascend310_infer/ || exit
    bash build.sh
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

    ascend310_infer/out/main $model ./preprocess_Result/00_data

}

function cal_acc()
{
    python postprocess.py
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
