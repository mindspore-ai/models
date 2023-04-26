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

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [NEED_PREPROCESS] [DEVICE_ID]
DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero.NEED_PREPROCESS is also 
optional, if you have done PREPROCESS, it is UNNECESSARY."
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
dataset=$2
video_path=$(get_real_path $3)
annotation_path=$(get_real_path $4)
device_id=0
need_preprocess='n'

if [ $# == 5 ]; then
    need_preprocess=$5
fi

if [ $# == 6 ]; then
    device_id=$6
    need_preprocess=$5
fi

echo "mindir name: "$model
echo "video path: "$video_path
echo "annotation path: "$annotation_path
echo "device id: "$device_id
echo "dataset: "$dataset
echo "need preprocess: " $need_preprocess

function pre_process_data()
{
    if [ -d pre_process_Result ]; then 
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --config_path ../${dataset}_config.yaml --video_path $video_path --annotation_path \
    $annotation_path --batch_size=1
}

function compile_app()
{
    cd ../ascend310_infer/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log
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
    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=./preprocess_Result/data --device_id=$device_id  &> infer.log
}

function cal_acc()
{
    python ../postprocess.py --config_path ../${dataset}_config.yaml --annotation_path $annotation_path &> acc.log &
}

if [ $need_preprocess == 'y' ]; then
    echo "Doing preprocess..."
    pre_process_data
fi

if [ $? -ne 0 ]; then 
    echo "execute preprocess failed"
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
