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


if [ $# != 6 ]
then 
    echo "Usage: bash run_eval.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH] [USE_G_FEATURE]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

MODEL_NAME=$1
DATASET_NAME=$2
DATASET_PATH=$(get_real_path $3)
CONFIG_PATH=$(get_real_path $4)
CHECKPOINT_PATH=$(get_real_path $5)
USE_G_FEATURE=$6

if [ $MODEL_NAME != "PCB" ] && [ $MODEL_NAME != "RPP" ]
then 
    echo "error: MODEL_NAME=$MODEL_NAME is invalid, please choose from ['PCB','RPP']"
exit 1
fi

if [ $DATASET_NAME != "market" ] && [ $DATASET_NAME != "duke" ] && [ $DATASET_NAME != "cuhk03" ]
then 
    echo "error: DATASET_NAME=$DATASET_NAME is invalid, please choose from ['market','duke','cuhk03']"
exit 1
fi

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi 

if [ ! -f $CONFIG_PATH ]
then 
    echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
exit 1
fi 

if [ ! -f $CHECKPOINT_PATH ]
then 
    echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file"
exit 1
fi 


script_path=$(readlink -f "$0")
script_dir_path=$(dirname "${script_path}")

LOG_SAVE_PATH=${script_dir_path}/output/log/${MODEL_NAME}/${DATASET_NAME}/eval/

if [ -d ${LOG_SAVE_PATH} ];
then
    rm -rf ${LOG_SAVE_PATH}
fi

python ${script_dir_path}/../eval.py --dataset_path=$DATASET_PATH --config_path=$CONFIG_PATH --checkpoint_file_path=$CHECKPOINT_PATH --use_G_feature=$USE_G_FEATURE --output_path ${script_dir_path}/output/
