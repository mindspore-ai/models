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

if [ $# != 8 ]; then
  echo "Usage: sh run_distribute_train.sh  [RANK_TABLE_FILE] [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] [MODALITY] [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

rank_table_file=$1
dataset_path=$(get_real_path $2)
dataset=$3
train_list_path=$(get_real_path $4)
train_list=$5
modality=$6
pretrained_path=$(get_real_path $7)
pretrained_path_name=$8


if [ ! -f $rank_table_file ]; then
  echo "error: RANK_TABLE_FILE=$rank_table_file is not a file"
  exit 1
fi

if [ ! -d $dataset_path ]; then
  echo "error: DATASET_PATH=$dataset_path is not a directory"
  exit 1
fi

if [ ! -d $pretrained_path ]; then
  echo "error: PRETRAINED_PATH=$pretrained_path is not a file"
  exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$rank_table_file
export RANK_START_ID=0

cd ..

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$((i + RANK_START_ID))
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp *.py ./train_parallel$i
  cp -r ./scripts ./train_parallel$i
  cp -r ./src ./train_parallel$i
  cp -r ./pre_trained ./train_parallel$i
  cd ./train_parallel$i || exit
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  python train.py --data_url=$dataset_path \
   --run_distribute=True \
   --dataset=$dataset \
   --train_list_path=$train_list_path \
   --train_list=$train_list \
   --modality=$modality \
   --pretrained_path=$pretrained_path \
   --pre_trained_name=$pretrained_path_name > log.txt 2>&1 &
  cd ..
done
