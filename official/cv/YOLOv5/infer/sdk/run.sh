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

pipeline_path=./config/yolov5.pipeline
dataset_path=../data/image/
ann_file=../data/instances_val2017.json
result_files=./result
# help message
if [[ $1 == --help || $1 == -h ]];then
    echo "usage:bash ./run.sh <args>"
    echo "parameter explain:
    --pipeline_path          set SDK infer pipeline, e.g. --pipeline_path=./config/yolov5.pipeline
    --dataset_path        root path of processed images, e.g. --dataset_path=../data/image
    --ann_file  the folder to save the semantic mask images, default: --ann_file=./result
    -h/--help           show help message
    "
    exit 1
fi

for para in "$@"
do
    if [[ $para == --pipeline_path* ]];then
        pipeline_path=`echo ${para#*=}`
    elif [[ $para == --dataset_path* ]];then
        dataset_path=`echo ${para#*=}`
    elif [[ $para == --ann_file* ]];then
        ann_file=`echo ${para#*=}`
    elif [[ $para == --result_files* ]];then
        result_files=`echo ${para#*=}`
    fi
done

if [[ $pipeline_path  == "" ]];then
   echo "[Error] para \"pipeline_path \" must be config"
   exit 1
fi
if [[ $dataset_path  == "" ]];then
   echo "[Error] para \"dataset_path \" must be config"
   exit 1
fi
if [[ $ann_file  == "" ]];then
   echo "[Error] para \"ann_file \" must be config"
   exit 1
fi

python3 main.py --pipeline_path=$pipeline_path \
                  --dataset_path=$dataset_path \
                  --ann_file=$ann_file \
                  --result_files=$result_files

exit 0
