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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID] [CATEGORY]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
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

echo "Mindir name: "$model
echo "dataset path: "$dataset_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id
echo "category: "$category

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


function preprocess_data() {
    if [ ! -d img ]; then
        mkdir ./img
    fi
    if [ -d img/$category ]; then
        rm -rf img/$category
    fi
    mkdir ./img/$category
    mkdir ./img/$category/pre
    mkdir ./img/$category/infer
    mkdir ./img/$category/label

    python ../preprocess.py \
    --data_dir $dataset_path \
    --img_dir ./img \
    --category $category
}

function compile_app() {
    cd ../ascend_310_infer || exit
    bash build.sh &> build.log
}

function infer() {
    cd - || exit
    if [ -d img/$category/result ]; then
        rm -rf img/$category/result
    fi
    mkdir img/$category/result
    mkdir img/$category/result/pre
    mkdir img/$category/result/infer

    if [ -d img/$category/time ]; then
        rm -rf img/$category/time
    fi
    mkdir img/$category/time
    mkdir img/$category/time/pre
    mkdir img/$category/time/infer

    ../ascend_310_infer/out/main \
        --mindir_path=$model \
        --input_path=./img/$category/pre \
        --result_path=./img/$category/result/pre \
        --time_path=./img/$category/time/pre \
        --device_id=$device_id &> pre_$category.log
    
    ../ascend_310_infer/out/main \
        --mindir_path=$model \
        --input_path=./img/$category/infer \
        --result_path=./img/$category/result/infer \
        --time_path=./img/$category/time/infer \
        --device_id=$device_id &> infer_$category.log
}

function cal_acc() {
    python ../postprocess.py \
        --result_dir ./img/$category/result/ \
        --img_dir ./img/$category/infer/ \
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
