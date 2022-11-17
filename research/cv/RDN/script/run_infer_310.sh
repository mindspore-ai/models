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
if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATASET_TYPE] [SCALE] [DEVICE_ID]"
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
dataset_type=$3
scale=$4


if [[ $scale -ne "2" &&  $scale -ne "3" &&  $scale -ne "4" ]]; then
    echo "[SCALE] should be in [2,3,4]"
exit 1
fi

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

log_file="./run_infer.log"
log_file=$(get_real_path $log_file)

echo "***************** param *****************"
echo "mindir name: "$model
echo "dataset path: "$data_path
echo "scale: "$scale
echo "log file: "$log_file
echo "***************** param *****************"

export PYTHONPATH=$PWD:$PYTHONPATH

function compile_app()
{
    echo "begin to compile app..."
    cd ../ascend310_infer || exit
    bash build.sh >> $log_file  2>&1
    cd -
    echo "finish compile app"
}

function preprocess()
{
    echo "begin to preprocess..."
    export DEVICE_ID=$device_id
    export RANK_SIZE=1
    rm -rf ../LR
    python ../preprocess.py --dataset_path=$data_path --dataset_type=$dataset_type --scale=$scale --save_path=../LR/ >> $log_file 2>&1
    echo "finish preprocess"
}

function infer()
{
    echo "begin to infer..."
    save_data_path=$data_path"/SR_bin/X"$scale
    if [ -d $save_data_path ]; then
        rm -rf $save_data_path
    fi
    mkdir -p $save_data_path
    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=../LR/ --device_id=$device_id --save_dir=$save_data_path >> $log_file 2>&1
    echo "finish infer"
}

function postprocess()
{
    echo "begin to postprocess..."
    export DEVICE_ID=$device_id
    export RANK_SIZE=1
    python ../postprocess.py --dataset_path=$data_path --dataset_type=$dataset_type --bin_path=$data_path"/SR_bin/X"$scale --scale=$scale  >> $log_file 2>&1
    echo "finish postprocess"
}

echo "" > $log_file
echo "read the log command: "
echo "    tail -f $log_file"

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed, check $log_file"
    exit 1
fi

preprocess
if [ $? -ne 0 ]; then
    echo "preprocess code failed, check $log_file"
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

cat $log_file | tail -n 3 | head -n 1
