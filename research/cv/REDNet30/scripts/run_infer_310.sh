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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [SAVE_BIN_PATH] [SAVE_OUTPUT_PATH] [DEVICE_ID](optional)
    DEVICE_ID can be set by environment variable device_id, otherwise the value is zero"
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
data_path=$(get_real_path $2)
save_bin_path=$(get_real_path $3)
save_output_path=$(get_real_path $4)


device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

log_file="./run_infer.log"
log_file=$log_file

echo "***************** param *****************"
echo "mindir name: "$model
echo "dataset path: "$data_path
echo "log file: "$log_file
echo "***************** param *****************"

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

export PYTHONPATH=$PWD:$PYTHONPATH

function preprocess()
{
    echo "waitting for preprocess finish..."
    python ./preprocess.py --dataset_path=$data_path --noise_path='./preprocess_Result/noise_data_310' --output_path='./preprocess_Result/data_310' >> $log_file 2>&1
    echo "preprocess finished!"
}

function compile_app()
{
    echo "begin to compile app..."
    cd ./ascend310_infer || exit
    bash build.sh >> $log_file  2>&1
    cd -
    echo "finish compile app"
}


function infer()
{
    echo "begin to infer..."
    if [ -d $save_bin_path ]; then
        rm -rf $save_bin_path
    fi
    mkdir -p $save_bin_path
    ./ascend310_infer/out/main --mindir_path=$model --dataset_path='./preprocess_Result/noise_data_310' --device_id=$device_id --save_dir=$save_bin_path >> $log_file 2>&1
    echo "finish infer"
}

function postprocess()
{
    echo "begin to postprocess..."
    export DEVICE_ID=$device_id
    export RANK_SIZE=1
    if [ -d $save_output_path ]; then
        rm -rf $save_output_path
    fi
    mkdir -p $save_output_path
    python ./postprocess.py --dataset_path='./preprocess_Result/data_310' --save_path=$save_output_path --bin_path=$save_bin_path >> $log_file 2>&1
    echo "finish postprocess"
}

preprocess
if [ $? -ne 0 ]; then
    echo "execute preprocess failed"
    exit 1
fi

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed, check $log_file"
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed, check $log_file"
    exit 1
fi

postprocess
if [ $? -ne 0 ]; then
    echo "postprocess failed, check $log_file"
    exit 1
fi
