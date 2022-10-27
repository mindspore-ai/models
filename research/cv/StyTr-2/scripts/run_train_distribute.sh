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
echo "bash scripts/run_train_distribute.sh  [BATCH_SIZE] [CONTENT_PATH] [STYLE_PATH] [AUXILIARY_PATH] [SAVE_PATH] [IMG_SAVE_PATH]"
echo "for example: bash scripts/run_train_distribute.sh 4 ./dataset/COCO2014/train2014 ./dataset/wikiart/train ./auxiliary ./save_model ./picture"
echo "=============================================================================================================="

BATCH_SIZE=$1
CONTENT_PATH=$2
STYLE_PATH=$3
AUXILIARY_PATH=$4
SAVE_PATH=$5
IMG_SAVE_PATH=$6

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

mpirun --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout -np 8 python train.py \
  --run_distribute True\
  --batch_size=$BATCH_SIZE \
  --content_dir=$CONTENT_PATH \
  --style_dir=$STYLE_PATH\
  --auxiliary_dir=$AUXILIARY_PATH\
  --save_dir=$SAVE_PATH\
  --save_picture=$IMG_SAVE_PATH\
  > output.train.log 2>&1 &