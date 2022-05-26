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
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_infer_gpu.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
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
device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi
echo "mindir name: "$model
echo "dataset path: "$dataset_path
echo "device id: "$device_id
export LD_PRELOAD=/root/anaconda3/lib/libpython3.7m.so
export PYTHONPATH=/home/mindspore/.local/lib/python3.7/site-packages
export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
function preprocess_data()
{
   if [ -d preprocess_Result ]; then
       rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python3.7 ../preprocess.py --data_path=$dataset_path
}
function compile_app()
{
    cd ../gpu_infer/ || exit
    bash build.sh &> build.log
}
function infer_train()
{
    cd - || exit
    if [ -d result_Files_train ]; then
        rm -rf ./result_Files_train
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files_train
    mkdir time_Result

    ../gpu_infer/out/main --mindir_path=$model --dataset_path=./preprocess_Result/train_data --device_id=$device_id --mode=train &> infer_train.log
}
function infer_test()
{
    if [ -d result_Files_test ]; then
        rm -rf ./result_Files_test
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files_test
    mkdir time_Result

    ../gpu_infer/out/main --mindir_path=$model --dataset_path=./preprocess_Result/test_data --device_id=$device_id --mode=test &> infer_test.log
}
function post_process()
{
    nohup python -u ../verify.py >> verify_log 2>&1 &
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
infer_train
if [ $? -ne 0 ]; then
    echo " execute inference train failed"
    exit 1
fi
infer_test
if [ $? -ne 0 ]; then
    echo " execute inference test failed"
    exit 1
fi
post_process
if [ $? -ne 0 ]; then
    echo " execute post_process failed"
    exit 1
fi