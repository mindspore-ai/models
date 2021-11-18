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

image_width=500
image_height=500
channel=3
sigma=15
# help message
if [[ $1 == --help || $1 == -h ]];then
    echo "usage:bash ./run.sh <args>"
    echo "parameter explain:
    --pipeline            set SDK infer pipeline, e.g. --pipeline=../data/config/brdnet.pipeline
    --clean_image_path    set the root path of image without noise, e.g. --clean_image_path=./Kodak24
    --image_width         set the resized image width, default: --image_width=500
    --image_height        set the resized image height, default: --image_height=500
    --channel             set the image channel, 3 for color, 1 for gray, default: --channel=3
    --sigma               set the level of noise, default: --sigma=15
    -h/--help             show help message
    "
    exit 1
fi

for para in "$@"
do
    if [[ $para == --pipeline* ]];then
        pipeline=`echo ${para#*=}`
    elif [[ $para == --clean_image_path* ]];then
        clean_image_path=`echo ${para#*=}`
    elif [[ $para == --image_width* ]];then
        image_width=`echo ${para#*=}`
    elif [[ $para == --image_height* ]];then
        image_height=`echo ${para#*=}`
    elif [[ $para == --channel* ]];then
        channel=`echo ${para#*=}`
    elif [[ $para == --sigma* ]];then
        sigma=`echo ${para#*=}`
    fi
done

if [[ $pipeline  == "" ]];then
   echo "[Error] para \"pipeline \" must be config"
   exit 1
fi
if [[ $clean_image_path  == "" ]];then
   echo "[Error] para \"clean_image_path \" must be config"
   exit 1
fi

python3 main.py --pipeline=$pipeline \
                  --clean_image_path=$clean_image_path \
                  --image_width=$image_width \
                  --image_height=$image_height \
                  --channel=$channel \
                  --sigma=$sigma

exit 0
