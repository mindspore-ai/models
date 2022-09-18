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
echo "bash run_standalone_train_gpu.sh DEVICE_ID VGG16_CKPT VAL_MINDRECORD_FILE"
echo "for example: bash run_standalone_train_gpu.sh 0 vgg16.ckpt val.mindrecord"
echo "=============================================================================================================="

DEVICE_ID=$1
VGG16=$2
VAL_MINDRECORD=$3

if [ $# -lt 3 ];
then
  echo "---------------------ERROR----------------------"
  echo "You must specify gpu device, vgg16 checkpoint and mindrecord file for valing"
  exit
fi

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
LOG_DIR=$PROJECT_DIR/../logs
if [ ! -d $LOG_DIR ]
then
  mkdir $LOG_DIR
fi

export DEVICE_ID=$DEVICE_ID
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python $PROJECT_DIR/../train.py \
    --device_target GPU \
    --epoches 100 \
    --lr 5e-4 \
    --basenet $VGG16 \
    --num_workers 2 \
    --val_mindrecord $VAL_MINDRECORD \
    --batch_size 4 > $LOG_DIR/training_gpu.log 2>&1 &

echo "The standalone train log is at /logs/training_gpu.log"
