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

if [ $# != 4 ]
then
    echo "Usage: bash run_standalone_train_regdb_i2v_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [RegDB_MODE] [DEVICE_ID]"
    echo "RegDB_MODE should be [i2v] or [v2i]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

RegDB_MODE=$3

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$4
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

TRAIN_LOG_PATH="train_regdb_"$RegDB_MODE
if [ -d $TRAIN_LOG_PATH ];
then
    rm -rf ./$TRAIN_LOG_PATH
fi
mkdir ./$TRAIN_LOG_PATH

cp ../*.py ./$TRAIN_LOG_PATH
cp -r ../src ./$TRAIN_LOG_PATH
cd ./$TRAIN_LOG_PATH || exit
env > env.log
echo "start training for device $DEVICE_ID"

python train.py \
--MSmode GRAPH_MODE \
--dataset RegDB \
--data_path $PATH1 \
--optim adam \
--lr 0.0035 \
--device_target GPU \
--gpu 0 $DEVICE_ID\
--pretrain $PATH2 \
--loss_func id+tri \
--trial 1 \
--regdb_mode $RegDB_MODE \
--epoch 80 \
--print_per_step 30 &> log &
cd ..