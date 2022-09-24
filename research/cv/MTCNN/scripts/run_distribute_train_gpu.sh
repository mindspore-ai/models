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
echo "bash run_distributed_train_gpu.sh MODEL_NAME(pnet|rnet|onet) DEVICE_NUM MINDRECORD_FILE"
echo "for example train PNet: bash run_distributed_train_gpu.sh pnet 8 pnet_train.mindrecord"
echo "=============================================================================================================="

MODEL=$1
DEVICE_NUM=$2
MINDRECORD_FILE=$3

if [ $# -lt 3 ];
then
  echo "---------------------ERROR----------------------"
  echo "You must specify model name and number of gpu devices and mindrecord file for training"
  exit
fi

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
LOG_DIR=$PROJECT_DIR/../logs
if [ ! -d $LOG_DIR ]
then
  mkdir $LOG_DIR
fi

if [ $MODEL == "pnet" ];
then
  END_EPOCH=32
elif [ $MODEL == 'rnet' ];
then
  END_EPOCH=24
else
  END_EPOCH=24
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

mpirun -n $DEVICE_NUM --allow-run-as-root python $PROJECT_DIR/../train.py \
    --distribute \
    --device_target GPU \
    --end_epoch $END_EPOCH \
    --model $MODEL \
    --mindrecord_file $MINDRECORD_FILE \
    --num_workers 8 \
    --save_ckpt_steps 100 > $LOG_DIR/distribute_training_gpu_$MODEL.log 2>&1 &

echo "The distributed train log is at /logs/distribute_training_gpu_$MODEL.log"
