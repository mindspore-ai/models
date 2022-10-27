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
if [ $# != 7 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_train.sh [BATCH_SIZE] [DEVICE_ID] [CONTENT_PATH] [STYLE_PATH] [AUXILIARY_PATH] [SAVE_PATH] [IMG_SAVE_PATH]"
  echo "for example: bash scripts/run_train.sh 4 0 ./dataset/COCO2014/train2014 ./dataset/wikiart/train ./auxiliary ./save_model ./picture "
  echo "=============================================================================================================="
exit 1
fi

BATCH_SIZE=$1
DEVICE_ID=$2
CONTENT_PATH=$3
STYLE_PATH=$4
AUXILIARY_PATH=$5
SAVE_PATH=$6
IMG_SAVE_PATH=$7

python train.py  \
  --batch_size=$BATCH_SIZE \
  --device_id=$DEVICE_ID \
  --content_dir=$CONTENT_PATH \
  --style_dir=$STYLE_PATH \
  --auxiliary_dir=$AUXILIARY_PATH \
  --save_dir=$SAVE_PATH \
  --save_picture=$IMG_SAVE_PATH \
  > train.log 2>&1 &