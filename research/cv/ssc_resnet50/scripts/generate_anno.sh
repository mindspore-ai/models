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
# used for generate one stage data
usage() {
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash generate_anno.sh DATA PERCENT ANNOTATION"
  echo "for example: bash generate_anno_first.sh  /path/to/dataset/train 15 /path/to/annotation.json"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
}

# used for generate one stage data
if [ $# -lt 3 ]; then
  usage
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

data_path=$(get_real_path $1)

python ../generate_anno.py \
       --data=$data_path \
       --percent=$2 \
       --annotation=$3
