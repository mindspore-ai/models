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

# The number of parameters transferred is not equal to the required number, print prompt information
if [ $# != 3 ]
then
  echo "==============================================================================================================="
  echo "Please run the script as: "
  echo "bash convert_model.sh [MODEL_NAME] [PTH_FILE] [MSP_FILE]"
  echo "for example: bash convert_model.sh data/resnet50-19c8e357.pth data/resnet50_msp.ckpt"
  echo "==============================================================================================================="
  exit 1
fi

# Get absolute path
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
MODEL_NAME=$1
PTH_FILE=$(get_real_path $2)
MSP_FILE=$(get_real_path $3)

cd $BASE_PATH/..
python pretrained_model_convert/pth_to_msp.py \
    --model=$MODEL_NAME \
    --pth_file="$PTH_FILE" \
    --msp_file="$MSP_FILE"
