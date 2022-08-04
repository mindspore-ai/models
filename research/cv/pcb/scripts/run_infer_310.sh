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


if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [USE_G_FEATURE][CONFIG_PATH] [DEVICE_ID](optional)
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
dataset_name=$2
dataset_path=$(get_real_path $3)
use_G_feature=$4
config_path=$(get_real_path $5)
query_image_path=./preprocess_Result/query/image
gallery_image_path=./preprocess_Result/gallery/image

device_id=0
if [ $# == 6 ]; then
    device_id=$6
fi

echo "mindir name: "$model
echo "dataset name: "$dataset_name
echo "dataset path: "$dataset_path
echo "use_G_feature: "$use_G_feature
echo "config path: "$config_path
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
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --dataset_name=$dataset_name --dataset_path=$dataset_path --config_path=$config_path --preprocess_result_path=./preprocess_Result &> preprocess.log
}


function compile_app()
{
    cd ../ascend310_infer || exit
    if [ -d out ]; then
        rm -rf ./out
    fi
    bash build.sh &> build.log
    cd - || exit
}

function infer()
{
    if [ -d query_result_files ]; then
        rm -rf ./query_result_files
    fi
    if [ -d gallery_result_files ]; then
        rm -rf ./gallery_result_files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir query_result_files
    mkdir gallery_result_files
    mkdir time_Result
    ../ascend310_infer/out/main --mindir_path=$model --query_image_path=$query_image_path --gallery_image_path=$gallery_image_path --device_id=$device_id  &> infer.log
}


function cal_metrics()
{
    python ../postprocess.py --preprocess_result_path=./preprocess_Result --query_prediction_path=./query_result_files --gallery_prediction_path=./gallery_result_files --use_G_feature=$use_G_feature --config_path=$config_path &> metrics.log
}


preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess data failed"
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
cal_metrics
if [ $? -ne 0 ]; then
    echo "calculate metrics failed"
    exit 1
fi
