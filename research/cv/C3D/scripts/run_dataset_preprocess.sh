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

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: bash run_dataset_preprocess.sh [DATASET_NAME] [RAR_FILE_PATH] [SPLIT_NUMBER]
    DATASET_NAME is the dataset name, it's value is 'HMDB51' or 'UCF101'.
    RAR_FILE_PATH is raw rar(or zip) file path
    SPLIT_NUMBER is the split number, it's value is '1', '2' or '3'."
exit 1
fi

if [ $1 != 'HMDB51' ] && [ $1 != 'UCF101' ]; then
    echo "DATASET_NAME must be 'HMDB51' or 'UCF101'."
exit 1
fi

if [ "${2:0:1}" != "/" ]; then
    echo "Please change $2 to an absolute path."
exit 1
fi

if [ ! -f $2 ]; then
    echo "dataset rar file path does not exist."
exit 1
fi

if [ $3 -ne 1 ] && [ $3 -ne 2 ] && [ $3 -ne 3 ]; then
    echo "SPLIT_NUMBER must be '1', '2' or '3'."
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

dataset_rar_file_path=$(get_real_path $2)

if [ ! -f ${dataset_rar_file_path} ]; then
    echo "Dataset rar file path does not exist!"
    exit 1
fi

python3.7  ../src/tools/dataset_preprocess.py $1 ${dataset_rar_file_path} $3
