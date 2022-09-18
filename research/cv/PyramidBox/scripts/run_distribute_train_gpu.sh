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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh DEVICE_NUM VGG16_CKPT VAL_MINDRECORD_FILE"
echo "for example: bash run_distribute_train_gpu.sh 8 vgg16.ckpt val.mindrecord"
echo "=============================================================================================================="

DEVICE_NUM=$1
VGG16=$2
VAL_MINDRECORD=$3

if [ $# -lt 3 ];
then
  echo "---------------------ERROR----------------------"
  echo "You must specify number of gpu devices, vgg16 checkpoint, mindrecord file for evaling"
  exit
fi

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
LOG_DIR=$PROJECT_DIR/../logs
if [ ! -d $LOG_DIR ]
then
  mkdir $LOG_DIR
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

mpirun -n $DEVICE_NUM --allow-run-as-root python $PROJECT_DIR/../train.py \
    --distribute True \
    --lr 5e-4 \
    --device_target GPU \
    --val_mindrecord $VAL_MINDRECORD \
    --epoches 100 \
    --basenet $VGG16 \
    --num_workers 1 \
    --batch_size 4 > $LOG_DIR/distribute_training_gpu.log 2>&1 &

echo "The distributed train log is at /logs/distribute_training_gpu.log"
