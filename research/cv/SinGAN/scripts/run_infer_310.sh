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
if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: bash scripts/run_infer_310.sh [MINDIR_PATH] [INPUT_DIR] [INPUT_NAME] [NOISE_AMP] [STOP_SCALE] [DEVICE_ID]
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
model_dir=$(get_real_path $1)
input_dir=$2
input_name=$3
noise_amp_dir=$4
stop_scale=$5
device_id=0
if [ $# == 6 ]; then
    device_id=$6
fi
echo "mindir dir path: "$model_dir
echo "input dir path: "$input_dir
echo "input image name: "$input_name
echo "noise_amp dir path: "$noise_amp_dir
echo "stop scale: "$stop_scale
echo "device id: "$device_id

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export PATH=$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

for((i=0; i<$stop_scale; i++))
do
    echo "scale num: "$i
    # preprocess
    if [ -d preprocess_Result ]; then
       rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python3.7 ./preprocess.py --output_path='./preprocess_Result/' --input_path='./postprocess_Result/' \
                                --input_dir=$input_dir --input_name=$input_name --scale_num=$i --noise_amp_path=$noise_amp_dir

    if [ $? -ne 0 ]; then
        echo "scale $i: preprocess dataset failed"
        exit 1
    fi

    # compile
    cd ./ascend310_infer/ || exit
    bash build.sh &> build.log
    if [ $? -ne 0 ]; then
        echo "scale $i: compile app code failed"
        exit 1
    fi

    # infer
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result

    ./ascend310_infer/out/main --mindir_path=$model_dir/$i/SinGAN.mindir --input0_path=./preprocess_Result/z_curr \
                                --input1_path=./preprocess_Result/I_prev --device_id=$device_id &> infer.log
    if [ $? -ne 0 ]; then
        echo "scale $i: execute inference failed"
        exit 1
    fi

    # postprocess
    if [ -d postprocess_Result ]; then
       rm -rf ./postprocess_Result
    fi
    mkdir postprocess_Result
    python3.7 ./postprocess.py --output_path='./postprocess_Result/' --input_dir=$input_dir \
                                --input_name=$input_name --scale_num=$i
    if [ $? -ne 0 ]; then
        echo "scale $i: execute post_process failed"
        exit 1
    fi
done
