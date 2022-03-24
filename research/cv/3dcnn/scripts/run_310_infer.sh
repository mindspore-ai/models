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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [TEST_PATH] [DEVICE_ID]
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
dataset_path=$(get_real_path $2)
test_path=$(get_real_path $3)
need_preprocess='y'
correction=True
device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "mindir name: "$model
echo "dataset path: "$dataset_path
echo "test path: "$test_path
echo "correction: "$correction
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

export RANK_SIZE=1
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
export ASCEND_HOME=/usr/local/Ascend
export PATH=$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/toolkit/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:/usr/local/Ascend/toolkit/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages
export PATH=/usr/local/python375/bin:$PATH
export NPU_HOST_LIB=/usr/local/Ascend/acllib/lib64/stub
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend
export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    mkdir preprocess_Result/flair_t2
    mkdir preprocess_Result/t1_t1ce
    python ../preprocess.py --data_path $dataset_path --test_path $test_path --correction $correction --pre_result_path ./preprocess_Result &> preprocess.log
}

function compile_app() {
    cd ../ascend_310_infer || exit
    bash build.sh &> build.log
}

function infer() {
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result

    ../ascend_310_infer/out/main --mindir_path=$model --input0_path=./preprocess_Result/flair_t2/ --input1_path=./preprocess_Result/t1_t1ce/ --device_id=$device_id &> infer.log
}

function cal_acc() {
    python ../postprocess.py --data_path $dataset_path --test_path $test_path --correction $correction --post_result_path ./result_Files &> acc.log
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
    echo "execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi