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

if [ $# != 3 ]; then
  echo "Usage: bash run_standalone_train_ascend.sh [TRAIN_DATA_DIR] [FILE_NAME] [VGG_MODEL]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATAPATH=$(get_real_path $1)
echo "info: load data from $DATAPATH"
FILENAME=$2
VGGPATH=$(get_real_path $3)

if [ ! -d $DATAPATH ]; then
  echo "error: TRAIN_DATA_DIR=$DATAPATH is not a directory"
  exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

echo "start training for device $DEVICE_ID"
env > env.log

nohup python -u ../csd_train.py \
      --device_target "Ascend" \
      --dir_data $DATAPATH \
      --filename $FILENAME \
      --lr 0.0001 \
      --epochs 1100 \
      --vgg_ckpt $VGGPATH \
      --contra_lambda 0 > train.log 2>&1 &