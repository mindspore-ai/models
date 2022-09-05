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

if [[ $# -lt 5 || $# -gt 5 ]]
then
    echo "Usage: bash run_eval_retrieval_images.sh [IMAGES_PATH] [GT_PATH] [RESULT_PATH] [PREPROCESS_RESULT] [BATCHSIZE]
    Please write the two images' name into the list_images.txt which you want to match.
    Usage example:bash run_eval_retrieval_images.sh ../data/ox ../data/ox_gt ./results/eval_features ./Preprocess_result 10
    "
exit 1
fi

num_worker=4
path_cur=$(dirname $0)
images_path=$1
gt_path=$2
result_path=$3
Preprocess_result=$4
batchsize=$5
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
    --resultnpz_path=$result_path \
    --batchsize=$batchsize 
}

function infer()
{
    ./delf ./Preprocess_result/images_batch/
}

function postprocess()
{
    python3 postprocess.py
}

function build_features_dataset()
{
    echo "Start to build features dataset..."
    python3 -u build_feature_dataset.py --ann_file="features.ann" \
    --index_features_dir="./results/eval_features" --image_path=$images_path \
    --gt_path=$gt_path --ann_path="./retrieval_dataset" &> build_features_dataset.log

}

function perform_retrieval()
{
    echo "Start to perform retrieval..."
    for((i=0;i<$num_worker;i++))
    do
        rm -rf ./retrieval_dataset/process$i
        mkdir -p ./retrieval_dataset/process$i
        echo "start process $i to retrieval images"
        python3 -u perform_retrieval.py --worker_id=$i \
        --worker_num=$num_worker --output_dir="./retrieval_dataset/process$i" \
        --ann_file="./retrieval_dataset/features.ann" --query_features_dir="./results/eval_features" \
        --index_features_dir="./results/eval_features" --image_path=$images_path \
        --gt_path=$gt_path --rank_file="ranks" > ./retrieval_dataset/process$i/retrieval$i.log 2>&1 &
    done
    wait
}

function calculate_mAP()
{
    echo "Start to calculate mAP..."
    python3 eval.py --ranks_path="./retrieval_dataset" --ranks_file="ranks" --worker_size=$num_worker\
    --image_path=$images_path --gt_path=$gt_path --output_dir="./" \
    --metric_name="mAP.txt" &> calculate_mAP.log

}
check_env
build_delf
cd ..

while [ 1 ]
do
if [ -d "./Preprocess_result" ]; then
    rm -r ./Preprocess_result
fi
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
rm -r ./Preprocess_result
rm -r ./results/result_Files
done
echo "successful"

# postprocess of retrieval
build_features_dataset
if [ $? -ne 0 ]; then
    echo "build features dataset failed"
    exit 1
fi

perform_retrieval
if [ $? -ne 0 ]; then
    echo "perform retrieval failed"
    exit 1
fi

calculate_mAP
if [ $? -ne 0 ]; then
    echo "calculate mAP failed"
    exit 1
fi

cat mAP.txt
