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
#Usage: bash train_distributed_gpu.sh [DATA_DIR] [SAVE_PATH] [RANK_SIZE] [EVAL_EACH_EPOCH] [PRETRAINED_CKPT_PATH](optional)

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "bash train_distributed_gpu.sh [DATA_DIR] [SAVE_PATH] [RANK_SIZE] [EVAL_EACH_EPOCH] [PRETRAINED_CKPT_PATH](optional)"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATA_DIR=$1
export RANK_SIZE=$3
SAVE_PATH=$2
EVAL_EACH_EPOCH=$4
PATH_CHECKPOINT=""
if [ $# == 5 ]; then
  PATH_CHECKPOINT=$5
fi

if [ -d "distribute_train_gpu" ]; then
  rm -rf ./distribute_train_gpu
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
mkdir ./distribute_train_gpu
cp -r $BASEPATH/../src ./distribute_train_gpu
cp $BASEPATH/../*.yaml ./distribute_train_gpu
cp $BASEPATH/../*.py ./distribute_train_gpu
cd ./distribute_train_gpu || exit

if [ $# == 4 ]; then
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  nohup python train.py  \
  --is_distributed=1 \
  --device_target=GPU \
  --ckpt_path=$SAVE_PATH \
  --eval_each_epoch=$EVAL_EACH_EPOCH\
  --train_data_dir=$DATA_DIR\
  --eval_data_dir= > log.txt 2>&1 &
fi

if [ $# == 5 ]; then
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  nohup python train.py  \
  --is_distributed=1 \
  --device_target=GPU \
  --ckpt_path=$SAVE_PATH \
  --eval_each_epoch=$EVAL_EACH_EPOCH\
  --pretrained=$PATH_CHECKPOINT \
  --train_data_dir=$DATA_DIR > log.txt 2>&1 &
fi

cd ../
