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

if [[ $# != 2 ]]; then
    echo "Usage:"
    echo "sh do_infer.sh [DATA_PATH] [DATA_LST]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -d $PATH1 ]
then
    echo "error: DATA_PATH=$PATH1 is not a directory."
    exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: DATA_LST=$PATH2 is not a file."
    exit 1
fi

rm -rf ./inferResults
mkdir ./inferResults

echo "Inference results will be stored in ./inferResults/."

python3 main.py --pipeline="../data/config/hrnetw48seg.pipeline" \
                  --data_path=$PATH1 \
                  --data_lst=$PATH2 \
                  --infer_result_path="./inferResults/"

echo "SDK inference task succeeded."

