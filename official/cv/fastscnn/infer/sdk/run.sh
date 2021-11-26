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

image_width=768
image_height=768
save_mask=1
mask_result_path=./mask_result
# help message
if [[ $1 == --help || $1 == -h ]];then
    echo "usage:bash ./run.sh <args>"
    echo "parameter explain:
    --pipeline          set SDK infer pipeline, e.g. --pipeline=../data/config/fastscnn.pipeline
    --image_path        root path of processed images, e.g. --image_path=../data/
    --image_width       set the image width,  default: --image_width=768
    --image_height      set the image height, default: --image_height=768
    --save_mask         whether to save the semantic mask images, 0 for False, 1 for True, default: --save_mask=1
    --mask_result_path  the folder to save the semantic mask images, default: --mask_result_path=./mask_result
    -h/--help           show help message
    "
    exit 1
fi

for para in "$@"
do
    if [[ $para == --pipeline* ]];then
        pipeline=`echo ${para#*=}`
    elif [[ $para == --image_path* ]];then
        image_path=`echo ${para#*=}`
    elif [[ $para == --image_width* ]];then
        image_width=`echo ${para#*=}`
    elif [[ $para == --image_height* ]];then
        image_height=`echo ${para#*=}`
    elif [[ $para == --save_mask* ]];then
        save_mask=`echo ${para#*=}`
    elif [[ $para == --mask_result_path* ]];then
        mask_result_path=`echo ${para#*=}`
    fi
done

if [[ $pipeline  == "" ]];then
   echo "[Error] para \"pipeline \" must be config"
   exit 1
fi
if [[ $image_path  == "" ]];then
   echo "[Error] para \"image_path \" must be config"
   exit 1
fi

python3 main.py --pipeline=$pipeline \
                  --image_path=$image_path \
                  --image_width=$image_width \
                  --image_height=$image_height \
                  --save_mask=$save_mask \
                  --mask_result_path=$mask_result_path

exit 0
