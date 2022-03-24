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
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [CONFIG_PATH] [SSIM_THRESHOLD] [L1_THRESHOLD] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero."
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
config_path=$(get_real_path $2)
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
need_preprocess="n"
ssim_threshold=$3
l1_threshold=$4

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi


echo "mindir name: "$model
echo "config path: "$config_path
echo "device id: "$device_id
echo "ssim threshold: "$ssim_threshold
echo "l1 threshold: "$l1_threshold

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
    cd $BASE_PATH/.. || exit
    if [ -d ./preprocess_result ]; then
        rm -rf ./preprocess_result
    fi
    python $BASE_PATH/../preprocess.py --config_path=$config_path  &> pre.log
}


function compile_app()
{
    cd $BASE_PATH/../ascend310_infer || exit
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


function infer()
{
    cd $BASE_PATH/.. || exit
    if [ -d ./postprocess_result ]; then
        rm -rf ./postprocess_result
    fi
    mkdir ./postprocess_result
    ./ascend310_infer/out/main --mindir_path=$model --dataset_path=./preprocess_result --device_id=$device_id --need_preprocess=$need_preprocess &> infer.log
}

function cal_acc()
{
    cd $BASE_PATH/.. || exit
    if [ -d save_img ]; then
        rm -rf ./save_img
    fi
    mkdir save_img
    python postprocess.py --config_path=$config_path --ssim_threshold=$ssim_threshold --l1_threshold=$l1_threshold --save_dir=save_img &> acc.log
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