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

if [[ $# -lt 2 || $# -gt 3 ]]
then
    echo "Usage: bash scripts/run_eval_match_images.sh [IMAGES_PATH] [CHECKPOINT] [DEVICES]
    DEVICES is optional.
    Please write the two images' name into the list_images.txt which you want to match.
    This shell surports that using multi Ascend card to extract feature.
    For example, if you want to use card 0 and card 3 to extract feature, just set DEVICES
    to be '03'. Others e.g. '012', '1'"
exit 1
fi

images_path=$1
checkpoint=$2
devices="0"
if [ $# == 3 ]
then
    devices=$3
fi

function extract_feature()
{
    echo "Start to extract features..."
    python src/extract_feature.py --devices=$devices --use_list_txt=True \
    --list_images_path="./list_images.txt" --ckpt_path=$checkpoint \
    --images_path=$images_path --target_path="./eval_features" &> extract_feature.log
}

function match_images()
{
    echo "Start to match images..."
    python src/match_images.py --list_images_path="./list_images.txt" \
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
