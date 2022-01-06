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

if [ $# != 5 ]; then
  echo "Usage: bash run_distribute_train_ascend.sh [TRAIN_DATA_DIR] [FILE_NAME] [TEACHER_MODEL] [VGG_MODEL] [RANK_TABLE_FILE]"
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
TPATH=$(get_real_path $3)
VGGPATH=$(get_real_path $4)
RANKTABLEPATH=$(get_real_path $5)

if [ ! -d $DATAPATH ]; then
  echo "error: TRAIN_DATA_DIR=$DATAPATH is not a directory"
  exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANKTABLEPATH

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp ../*.py ./train_parallel$i
  cp *.sh ./train_parallel$i
  cp -r ../src ./train_parallel$i
  cd ./train_parallel$i || exit
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  env >env.log
  nohup python -u csd_train.py \
        --device_target "Ascend" \
        --dir_data $DATAPATH \
        --filename $FILENAME \
        --lr 0.0001 \
        --epochs 1100 \
        --ckpt_path $TPATH \
        --vgg_ckpt $VGGPATH \
        --contra_lambda 200 > train.log 2>&1 &
  cd ..
done

