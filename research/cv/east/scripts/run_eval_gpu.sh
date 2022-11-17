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
if [ $# != 3 ]
then
    echo "Usage: bash run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID]"
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
CKPT_PATH=$(get_real_path $2)
echo $DATASET_PATH
echo $CKPT_PATH

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if [ ! -f $CKPT_PATH ]
then
    echo "error: CKPT_PATH=$CKPT_PATH is not a file"
exit 1
fi

export DEVICE_NUM=1
export CUDA_VISIBLE_DEVICES=$3

echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log
python eval.py \
    --device_target=GPU \
    --test_img_path=$DATASET_PATH \
    --checkpoint_path=$CKPT_PATH \
    --device_num=0 > log.txt 2>&1 &
