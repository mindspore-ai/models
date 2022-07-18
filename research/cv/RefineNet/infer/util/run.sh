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

output_path=../mxbase/bin
# help message
if [[ $1 == --help || $1 == -h ]];then
    echo "usage:bash ./run.sh <args>"
    echo "parameter explain:
    --data_root             dataset root path, e.g. --data_root=../data
    --data_lst              image path file, e.g. --data_lst=../data/voc_val_lst.txt
    --output_path           bin file path, default: --output_path=../mxbase/bin
    -h/--help               show help message
    "
    exit 1
fi

for para in "$@"
do
    if [[ $para == --data_root* ]];then
        data_root=`echo ${para#*=}`
    elif [[ $para == --data_lst* ]];then
        data_lst=`echo ${para#*=}`
    elif [[ $para == --output_path* ]];then
        output_path=`echo ${para#*=}`
    fi
done

if [[ $data_root  == "" ]];then
   echo "[Error] para \"data_root\" must be config"
   exit 1
fi
if [[ $data_lst  == "" ]];then
   echo "[Error] para \"data_lst\" must be config"
   exit 1
fi

python3 preprocess.py --data_root=$data_root \
                            --data_lst=$data_lst \
                            --output_path=$output_path

exit 0
