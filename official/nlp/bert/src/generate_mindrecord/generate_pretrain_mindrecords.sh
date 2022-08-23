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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash ./generate_pretrain_mindrecords.sh INPUT_FILES_PATH OUTPUT_FILES_PATH VOCAB_FILE"
echo "for example: bash ./generate_pretrain_mindrecords.sh ./wiki-clean-aa ./output/ ./bert-base-uncased-vocab.txt"
echo "=============================================================================================================="

if [ $# -ne 3 ]
then
    echo "Usage: bash ./generate_pretrain_mindrecords.sh INPUT_FILES_PATH OUTPUT_FILES_PATH VOCAB_FILE"
exit 1
fi

get_real_path(){
    if [ -z "$1" ]; then
        echo ""
    elif [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

run_create_mindrecords(){
    rm -rf ./logs
    mkdir ./logs
    for file in $1/*
    do
        [[ -e "$file" ]]
        input_file_name=$file
        echo "============input_file_name is $input_file_name"
        file_name=${input_file_name##*/}
        output_file_name=$2/$file_name.mindrecord
        echo "============output_file_name is $output_file_name"
        echo "============vocab_file name is $3"
        python generate_pretrain_mindrecord.py --input_file $input_file_name \
                                               --output_file $output_file_name \
                                               --vocab_file $3 &> ./logs/$file_name.txt &
    done
}

input_files_path=$(get_real_path $1)
output_files_path=$(get_real_path $2)
vocab_file=$(get_real_path $3)
run_create_mindrecords $input_files_path $output_files_path $vocab_file
