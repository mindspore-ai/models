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

if [ $# != 5 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_distribute_train_ascend.sh [TRAIN_PATH] [DEVICE_NUM] [EPOCH_SIZE] [CONFIG_PATH] [RANK_TABLE_FILE]"
  echo "for example: bash run_distribute_train_ascend.sh ../train.py 8 279 ../default_config.yaml ../rank_table_8pcs.json "
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
TRAIN_PATH=$(get_real_path $1)
CONFIG_PATH=$(get_real_path $4)
RANK_TABLE_FILE=$(get_real_path $5)

echo $TRAIN_PATH
echo $CONFIG_PATH
echo $RANK_TABLE_FILE

if [ ! -f $TRAIN_PATH ]; then
  echo "error: TRAIN_PATH=$TRAIN_PATH is not a file"
  exit 1
fi

if [ ! -f $CONFIG_PATH ]; then
  echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
  exit 1
fi

if [ ! -f $RANK_TABLE_FILE ]; then
  echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
  exit 1
fi
ulimit -u unlimited
export RANK_SIZE=$2
export EPOCH_SIZE=$3
export HCCL_CONNECT_TIMEOUT=6000
export RANK_TABLE_FILE=$RANK_TABLE_FILE
echo $RANK_SIZE

for ((i = 0; i <= $RANK_SIZE - 1; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp ../*.py ./train_parallel$i
  cp ../*.yaml ./train_parallel$i
  cp -r ../src ./train_parallel$i
  cd ./train_parallel$i || exit
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  env >env.log
  python train.py \
    --config_path=$CONFIG_PATH \
    --distribute="true" \
    --device_target="Ascend" \
    --epoch_size=$EPOCH_SIZE \
    --device_num=$RANK_SIZE \
    --enable_save_ckpt="true" \
    --enable_lossscale="true" \
    --do_shuffle="true" \
    --checkpoint_path="" \
    --save_checkpoint_num=30 > log.txt 2>&1 &
  cd ..
done
