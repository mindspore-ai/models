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

if [[ $# -lt 1 || $# -gt 2 ]]
then
    echo "
    Please write the two images' name into the list_images.txt whih you want to match.
    Usage example: bash run_eval_match_images.sh ../data/ox 
    "
exit 1
fi
path_cur=$(dirname $0)
images_path=$1
Preprocess_result="./Preprocess_result"
result_path="./results/eval_features"
if [ $# == 2 ]
then
    list_images_path=$2
else
    list_images_path="./list_images.txt"
fi
function check_env()
{
    echo "-------------------------"
    echo "--${OPENSOURCE_DIR}--"
    # set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user    
    if [ ! "${ASCEND_VERSION}" ]; then
        export ASCEND_VERSION=ascend-toolkit/latest
        echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
    else
        echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
    fi

    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to ./ when it was not specified by user
        export ARCH_PATTERN=./
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user"
    fi
}

function build_delf()
{
    cd $path_cur
    rm -rf build
    mkdir -p build
    rm -rf results/result_Files
    mkdir -p results/result_Files
    cd build
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
        echo "Failed to build delf."
        exit ${ret}
    fi
    make install
}

function preprocess()
{
    python3 preprocess.py  --images_path=$images_path --pre_path=$Preprocess_result\
    --resultnpz_path=$result_path --use_list_txt=True \
    --batchsize="2" &> preprocess.log
}

function infer()
{
    ./delf ./Preprocess_result/images_batch/
}

function postprocess()
{
    python3 postprocess.py
}

function match_images()
{
    echo "Start to match images..."
    python3 match_images.py --list_images_path=$list_images_path \
    --images_path=$images_path --feature_path="./results/eval_features" \
    --output_image="eval_match.png" &> match_images.log
}

check_env
build_delf
cd ..
# infer
while [ 1 ]
do
rm -rf ./Preprocess_result
preprocess
if [ $? -ne 0 ]; then
    exit 1
fi
if [ ! -d "./Preprocess_result/images_batch" ]; then
    break
fi
echo "aaaa"
if [ -d "./results/result_Files" ]; then
        rm -r ./results/result_Files
fi
mkdir ./results/result_Files
infer
if [ $? -ne 0 ]; then
    exit 1
fi
postprocess
if [ $? -ne 0 ]; then
    exit 1
fi
rm -rf ./Preprocess_result
rm -r ./results/result_Files
done
echo "successful"
#match
match_images
if [ $? -ne 0 ]; then
    echo "match images failed"
    exit 1
fi
