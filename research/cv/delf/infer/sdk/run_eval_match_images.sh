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

if [[ $# -lt 2 || $# -gt 2 ]]
then
    echo "Usage: bash run_eval_match_images.sh [IMAGES_PATH] [PIPELINE_PATH]
    Please write the two images' name into the list_images.txt which you want to match.
    Usage example: bash run_eval_match_images.sh ../data/ox ./delf.pipeline
    "
exit 1
fi

images_path=$1
pipeline_path=$2

function extract_feature()
{
    echo "Start to extract features..."
    python3 main.py --use_list_txt=True \
    --list_images_path="./list_images.txt" --PL_PATH=$pipeline_path \
    --images_path=$images_path --target_path="./eval_features" &> extract_feature.log
}

function match_images()
{
    echo "Start to match images..."
    python3 match_images.py --list_images_path="./list_images.txt" \
    --images_path=$images_path --feature_path="./eval_features" \
    --output_image="eval_match.png" &> match_images.log
}

extract_feature
if [ $? -ne 0 ]; then
    echo "extract feature failed"
    exit 1
fi

match_images
if [ $? -ne 0 ]; then
    echo "match images failed"
    exit 1
fi
