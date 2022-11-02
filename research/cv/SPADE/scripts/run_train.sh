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

set -e
if [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash ./scripts/run_train.sh [vgg_ckpt_path] [data_root] [load_epoch]"
    echo "bash ./scripts/run_train.sh [vgg_ckpt_path] [data_root] [load_epoch]"
    echo "For example: bash ./scripts/run_train.sh ./vgg/vgg19.ckpt ./ADEChallengeData2016 0"
    echo "If load_epoch is equal to 0, checkpoint is not loaded, otherwise round [load_epoch] checkpoint will be loaded "
    echo "================================================================================================================="
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
VGG_CKPT_PATH=$(get_real_path $1)
DATA_ROOT=$(get_real_path $2)
LOAD_EPOCH=$3

python -u ./train.py --distribute False \
                      --batchSize 1 \
                      --vgg_ckpt_path $VGG_CKPT_PATH \
                      --dataroot $DATA_ROOT \
                      --now_epoch $LOAD_EPOCH > train.log 2>&1 &