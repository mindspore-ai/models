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

if [[ $# -lt 3 || $# -gt 4 ]]
then
    echo "Usage: bash scripts/run_eval_retrieval_images.sh [IMAGES_PATH] [GT_PATH] [CHECKPOINT] [DEVICES]
    DEVICES is optional.
    Please write the two images' name into the list_images.txt which you want to match.
    This shell surports that using multi Ascend card to extract feature.
    For example, if you want to use card 0 and card 3 to extract feature, just set DEVICES
    to be '03'. Others e.g. '012', '1'"
exit 1
fi

num_worker=4
images_path=$1
gt_path=$2
checkpoint=$3
devices="0"
if [ $# == 4 ]
then
    devices=$4
fi

function extract_feature()
{
    echo "Start to extract features..."
    python src/extract_feature.py --devices=$devices --use_list_txt=False \
    --ckpt_path=$checkpoint \
    --images_path=$images_path --target_path="./eval_features" &> extract_feature.log
}

function build_features_dataset()
{
    echo "Start to build features dataset..."
    python -u src/build_feature_dataset.py --ann_file="features.ann" \
    --index_features_dir="./eval_features" --image_path=$images_path \
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
        python -u src/perform_retrieval.py --worker_id=$i \
        --worker_num=$num_worker --output_dir="./retrieval_dataset/process$i" \
        --ann_file="./retrieval_dataset/features.ann" --query_features_dir="./eval_features" \
        --index_features_dir="./eval_features" --image_path=$images_path \
        --gt_path=$gt_path --rank_file="ranks" > ./retrieval_dataset/process$i/retrieval$i.log 2>&1 &
    done
    wait

}

function calculate_mAP()
{
    echo "Start to calculate mAP..."
    python eval.py --ranks_path="./retrieval_dataset" --ranks_file="ranks" --worker_size=$num_worker\
    --image_path=$images_path --gt_path=$gt_path --output_dir="./" \
    --metric_name="mAP.txt" &> calculate_mAP.log

}

extract_feature
if [ $? -ne 0 ]; then
    echo "extract feature failed"
    exit 1
fi

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
