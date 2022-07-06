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
if [ $# != 6 ] && [ $# != 7 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]"
    echo " or "
    echo "bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH] [OUTPUT_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CFG_PATH=$(get_real_path $2)
CKPT_PATH=$(get_real_path $3)
IMAGES_PATH=$(get_real_path $4)
MASKS_PATH=$(get_real_path $5)
ANNO_PATH=$(get_real_path $6)
OUTPUT_PATH=$(get_real_path $7)

if [ ! -f $CKPT_PATH ]
then
    echo "Error: CKPT_PATH=$CKPT_PATH is not a file."
exit 1
fi

if [ ! -f $ANNO_PATH ]
    then
        echo "Error: ANNO_PATH=$ANNO_PATH is not a file!"
    exit 1
fi

if [ ! -d $MASKS_PATH ]
    then
        echo "Error: MASKS_PATH=$MASKS_PATH is not a dir!"
    exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$1

if [ ! -d "./logs" ]
then
  mkdir "./logs"
fi

echo "Start evaluation for device $DEVICE_ID"
if [ $# == 6 ]
then
  python eval.py \
    --checkpoint_path=$CKPT_PATH \
    --device_target='GPU' \
    --device_num=$DEVICE_NUM \
    --device_id=$DEVICE_ID \
    --data_root=$IMAGES_PATH \
    --anno_path=$ANNO_PATH \
    --config_path=$CFG_PATH \
    --eval_masks_root=$MASKS_PATH > ./logs/eval_log.txt 2>&1 &
fi
if [ $# == 7 ]
then
  python eval.py \
    --checkpoint_path=$CKPT_PATH \
    --device_target='GPU' \
    --device_num=$DEVICE_NUM \
    --device_id=$DEVICE_ID \
    --data_root=$IMAGES_PATH \
    --anno_path=$ANNO_PATH \
    --eval_masks_root=$MASKS_PATH \
    --config_path=$CFG_PATH \
    --output_path=$OUTPUT_PATH > ./logs/eval_log.txt 2>&1 &
fi
