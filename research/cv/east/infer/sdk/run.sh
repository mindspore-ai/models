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

pipeline=./pipeline/east.pipeline
image_path=../data/image/
result_path=./result
# help message
if [[ $1 == --help || $1 == -h ]];then
    echo "usage:bash ./run.sh <args>"
    echo "parameter explain:
    --pipeline          set SDK infer pipeline, e.g. --pipeline=../data/config/fastscnn.pipeline
    --image_path        root path of processed images, e.g. --image_path=../data/image
    --result_path  the folder to save the semantic mask images, default: --mask_result_path=./result
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
    elif [[ $para == --result_path* ]];then
        result_path=`echo ${para#*=}`
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

python3.7 main.py --pipeline=$pipeline \
                  --image_path=$image_path \
                  --result_path=$result_path

exit 0
