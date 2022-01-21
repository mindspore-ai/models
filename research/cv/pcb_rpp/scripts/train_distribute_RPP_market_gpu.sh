#!/bin/bash
# Copyright 2020-2022 Huawei Technologies Co., Ltd
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


if [ $# != 3 ] && [ $# != 4 ]
then 
    echo "Usage: bash train_distribute_RPP_market.sh [RANK_SIZE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)"
    exit 1
fi


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $3)

if [ $# == 4 ]; then
  PRETRAINED_CKPT_PATH=$(get_real_path $4)
else
  PRETRAINED_CKPT_PATH=""
fi

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if [ ! -d $CONFIG_PATH ]
then 
    echo "error: CONFIG_PATH=$CONFIG_PATH is not a directory"
exit 1
fi

if [ $# == 4 ] && [ ! -f $PRETRAINED_CKPT_PATH ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PRETRAINED_CKPT_PATH is not a file"
exit 1
fi

export RANK_SIZE=$1

script_path=$(readlink -f "$0")
script_dir_path=$(dirname "${script_path}")

LOG_SAVE_PATH=${script_dir_path}/output/log/RPP/market/train/
CHECKPOINT_SAVE_PATH=${script_dir_path}/output/checkpoint/RPP/market/train/
if [ -d $LOG_SAVE_PATH ];
then
    rm -rf $LOG_SAVE_PATH
fi

if [ -d $CHECKPOINT_SAVE_PATH ];
then
    rm -rf $CHECKPOINT_SAVE_PATH
fi

mpirun -n $RANK_SIZE --output-filename log_output --allow-run-as-root --merge-stderr-to-stdout \
  python ${script_dir_path}/../train.py \
  --dataset_path=$DATASET_PATH \
  --config_path=${CONFIG_PATH}/train_PCB.yaml \
  --checkpoint_file_path=$PRETRAINED_CKPT_PATH \
  --device_num=$RANK_SIZE \
  --run_distribute=True \
  --output_path ${script_dir_path}/output/ \
  --batch_size 32 \
  --learning_rate 0.4 \
  --device_target GPU > ../output.train_pcb.log 2>&1

mpirun -n $RANK_SIZE --output-filename log_output --allow-run-as-root --merge-stderr-to-stdout \
  python ${script_dir_path}/../train.py \
  --dataset_path=$DATASET_PATH \
  --config_path=${CONFIG_PATH}/train_RPP.yaml \
  --checkpoint_file_path=${CHECKPOINT_SAVE_PATH}/ckpt_0/PCB-20_50.ckpt \
  --device_num=$RANK_SIZE \
  --run_distribute=True \
  --output_path ${script_dir_path}/output/ \
  --batch_size 32 \
  --learning_rate 0.04 \
  --device_target GPU > ../output.train_rpp.log 2>&1 &
