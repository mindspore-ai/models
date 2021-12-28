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

if [ $# != 5 ]
then 
    echo "Usage: bash run_process_data.sh [POIND_CLOUD_PATH] [OUTPUT_PATH] [MIN_ID] [MAX_ID] [MODE]"
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
if [ ! -d $PATH1 ]
then 
    echo "error: POIND_CLOUD_PATH=$PATH1 is not a directory"
exit 1
fi

input_path=$1
output_path=$2
min_file=$3
max_file=$4
mode=$5

python ../process_data.py --input_route=$input_path --output_route=$output_path --min_file=$min_file --max_file=$max_file --mode=$mode
