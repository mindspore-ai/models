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

image_width=1024
image_height=512
output_path=./bin
# help message
if [[ $1 == --help || $1 == -h ]];then
    echo "usage:bash ./preprocess.sh <args>"
    echo "parameter explain:
    --image_path       root path of processed images, e.g. --image_path=../data/
    --image_width     set the image width, default: --image_width=1024
    --image_height    set the image height, default: --image_height=512
    --output_path      bin file path, default: --output_path=./bin
    -h/--help             show help message
    "
    exit 1
fi

for para in "$@"
do
    if [[ $para == --image_path* ]];then
        image_path=`echo ${para#*=}`
    elif [[ $para == --image_width* ]];then
        image_width=`echo ${para#*=}`
    elif [[ $para == --image_height* ]];then
        image_height=`echo ${para#*=}`
    elif [[ $para == --output_path* ]];then
        output_path=`echo ${para#*=}`
    fi
done

if [[ $image_path  == "" ]];then
   echo "[Error] para \"image_path \" must be config"
   exit 1
fi

python3 main.py --image_path=$image_path \
                  --image_width=$image_width \
                  --image_height=$image_height \
                  --output_path=$output_path

exit 0
