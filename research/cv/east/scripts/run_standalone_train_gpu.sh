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
    echo "Usage: bash run_distribute_train_gpu.sh [DATASET_PATH] [PRETRAINED_BACKBONE] [DEVICE_ID]"
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
PRETRAINED_BACKBONE=$(get_real_path $2)
RANK_TABLE_FILE=$(get_real_path $3)
echo $DATASET_PATH
echo $PRETRAINED_BACKBONE
echo $RANK_TABLE_FILE

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if [ ! -f $PRETRAINED_BACKBONE ]
then
    echo "error: PRETRAINED_PATH=$PRETRAINED_BACKBONE is not a file"
exit 1
fi

export CUDA_VISIBLE_DEVICES=$3


echo "start training for device $DEVICE_ID"
python train.py \
    --device_target=GPU \
    --data_dir=$DATASET_PATH \
    --pretrained_backbone=$PRETRAINED_BACKBONE \
    --device_id=0 \
    --max_epoch=400 \
    --is_distributed=0 > log.txt 2>&1 &
