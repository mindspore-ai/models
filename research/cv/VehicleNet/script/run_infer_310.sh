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

if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "Usage: bash run_infer_310.sh [TEST_BIN_PATH] [QUERY_BIN_PATH] [MINDIR_PATH] [TEST_DATA_PATH] [QUERY_DATA_PATH] [TEST_LABEL] [QUERY_LABEL] [TEST_OUT_PATH] [QUERY_OUT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

test_bin_path=$(get_real_path $1)
query_bin_path=$(get_real_path $2)
model=$(get_real_path $3)
input1_path=$(get_real_path $4)
input2_path=$(get_real_path $5)
test_label=$(get_real_path $6)
query_label=$(get_real_path $7)
test_out_path=$(get_real_path $8)
query_out_path=$(get_real_path $9)

device_id=0
if [ $# == 10 ]; then
    device_id=${10}
fi

echo "mindir name: "$model
echo "input1 path: "$input1_path
echo "input2 path: "$input2_path
echo "device id: "$device_id

export ASCEND_HOME=/usr/local/Ascend
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

function preprocess()
{
    cd ../ || exit
    if [ -d data ]; then
        rm -rf ./data
    fi
    mkdir data
    cd ./data
    mkdir test
    mkdir query
    cd ../scripts
    python ../preprocess.py --test_bin_path $test_bin_path --query_bin_path $query_bin_path --test_path $input1_path --query_path $input2_path
}

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function test_infer()
{
    cd - || exit
    if [ -d test_result_Files ]; then
        rm -rf ./test_result_Files
    fi
    if [ -d test_time_Result ]; then
        rm -rf ./test_time_Result
    fi
    mkdir test_result_Files
    mkdir test_time_Result
    input_type="test"
    ../ascend310_infer/out/main --mindir_path=$model --input0_path=$input1_path --input_type=$input_type --device_id=$device_id &> test_infer.log
}

function query_infer()
{
    if [ -d query_result_Files ]; then
        rm -rf ./query_result_Files
    fi
    if [ -d query_time_Result ]; then
        rm -rf ./query_time_Result
    fi
    mkdir query_result_Files
    mkdir query_time_Result
    input_type="query"
    ../ascend310_infer/out/main --mindir_path=$model --input0_path=$input2_path --input_type=$input_type --device_id=$device_id &> query_infer.log
}

function cal_acc()
{
    python ../postprocess.py --test_label $test_label --query_label $query_label --test_out_path $test_out_path --query_out_path $query_out_path &> acc.log
}

preprocess
if [ $? -ne 0 ]; then
    echo "preprocess failed"
    exit 1
fi

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

test_infer
if [ $? -ne 0 ]; then
    echo " execute test inference failed"
    exit 1
fi

query_infer
if [ $? -ne 0 ]; then
    echo " execute query inference failed"
    exit 1
fi

cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
