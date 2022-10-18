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
    echo "Usage: bash run_infer_310.sh [GEN_MINDIR_PATH] [IMAGES_PATH] [GT_PATH] [DEVICE_ID]
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

gen_model=$(get_real_path $1)
data_path=$(get_real_path $2)
gt_path=$(get_real_path $3)

num_worker=4

device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "generator mindir name: "$gen_model
echo "dataset path: "$data_path
echo "device id: "$device_id

function preprocess_data()
{
    rm -rf ./preprocess_images
    rm -rf ./image_pyramids
    echo "Start to preprocess images..."
    python -u ./src/preprocess.py --use_list_txt="False" --images_path=$data_path --output_path="./preprocess_images" --size_path="./image_pyramids" >preprocess.log 2>&1 &
}

function compile_app()
{
    echo "Start to compile source code..."
    cd ./ascend310_infer || exit
    bash build.sh &> build.log
    echo "Compile successfully."
    cd ..
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    echo "Start to execute inference..."
    ./ascend310_infer/out/main --gen_mindir_path=$gen_model --dataset_path="./preprocess_images" \
    --device_id=$device_id >infer.log 2>&1 &
}

function postprocess_data()
{
    rm -rf ./feature_Files
    echo "Start to postprocess image file..."
    python ./src/postprocess.py --use_list_txt="False" \
    --bin_path="./result_Files/" --target_path="./feature_Files/" --images_path=$data_path \
    --size_path="./image_pyramids" >postprocess.log 2>&1 &
}

function build_features_dataset()
{
    echo "Start to build features dataset..."
    python -u ./src/build_feature_dataset.py --ann_file="features.ann" \
    --index_features_dir="./feature_Files" --image_path=$data_path \
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
        python -u ./src/perform_retrieval.py --worker_id=$i \
        --worker_num=$num_worker --output_dir="./retrieval_dataset/process$i" \
        --ann_file="./retrieval_dataset/features.ann" --query_features_dir="./feature_Files" \
        --index_features_dir="./feature_Files" --image_path=$data_path \
        --gt_path=$gt_path --rank_file="ranks" > ./retrieval_dataset/process$i/retrieval$i.log 2>&1 &
    done
    wait

}

function calculate_mAP()
{
    echo "Start to calculate mAP..."
    python ./eval.py --ranks_path="./retrieval_dataset" --ranks_file="ranks" --worker_size=$num_worker\
    --image_path=$data_path --gt_path=$gt_path --output_dir="./" \
    --metric_name="mAP.txt" &> calculate_mAP.log

}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

preprocess_data
infer
postprocess_data
wait
echo "Infer successfully."


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
