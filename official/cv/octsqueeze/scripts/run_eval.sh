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
    echo "Usage: bash run_eval.sh [TEST_DATASET_PATH] [COMPRESSED_DATA_PATH] [RECONSTRUCTED_DATA_PATH] [MODE] [DEVICE]"
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
    echo "error: TEST_DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

PATH4=$(get_real_path $4)
if [ ! -f $PATH4 ]
then
    echo "error: MODE=$PATH4 is not a file"
exit 1
fi

test_dataset=$1
compression=$2
recon=$3
model=$4
device_target=$5

python ../eval.py --test_dataset=$test_dataset --compression=$compression --recon=$recon --model=$model --device_target=$device_target

