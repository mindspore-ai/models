#!/bin/bash
# Copyright 2020-2022 Huawei Technologies Co., Ltd
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


if [ $# != 4 ]
then 
    echo "Usage: bash eval_RPP_market.sh [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH] [USE_G_FEATURE]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
CONFIG_PATH=$(get_real_path $2)
CHECKPOINT_PATH=$(get_real_path $3)
USE_G_FEATURE=$4

if [ ! -d "$DATASET_PATH" ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi 

if [ ! -f "$CONFIG_PATH" ]
then 
    echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
exit 1
fi 

if [ ! -f "$CHECKPOINT_PATH" ]
then 
    echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file"
exit 1
fi 


script_path=$(readlink -f "$0")
script_dir_path=$(dirname "${script_path}")

LOG_SAVE_PATH=${script_dir_path}/output/log/RPP/market/eval/
if [ -d "$LOG_SAVE_PATH" ];
then
    rm -rf "$LOG_SAVE_PATH"
fi

python "${script_dir_path}"/../eval.py --dataset_path="$DATASET_PATH" --config_path="$CONFIG_PATH" --checkpoint_file_path="$CHECKPOINT_PATH" --use_G_feature="$USE_G_FEATURE" --output_path "${script_dir_path}"/output/ --device_target GPU > ../output.eval.log 2>&1 &


