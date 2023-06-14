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

if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_standalone_train_gpu.sh [DATASET_PATH] [BACKONE_PATH] [CATEGORY] [DEVICE_ID]"
    echo "For example: bash scripts/run_standalone_train_gpu.sh ./data ./stpm_backbone.ckpt zipper 0"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi
set -e

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_PATH=$(get_real_path $1)
CKPT_PATH=$(get_real_path $2)
CATEGORY=$3
DEVICE_ID=$4

TRAIN_PATH=train_$CATEGORY
if [ -d $TRAIN_PATH ];
then
    rm -rf ./$TRAIN_PATH
fi
mkdir ./$TRAIN_PATH
cd ./$TRAIN_PATH
env > env0.log
echo "[INFO] start train dataset $CATEGORY with device_id: $DEVICE_ID"
python ../train.py \
        --device_target "GPU" \
        --batch_size 24 \
        --epoch 100 \
        --lr 0.3 \
        --dataset_path "$DATA_PATH" \
        --pre_ckpt_path "$CKPT_PATH" \
        --category "$CATEGORY" \
        --device_id $DEVICE_ID \
        > train.log 2>&1 &
cd ../
