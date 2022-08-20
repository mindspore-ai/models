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
echo "bash run_standalone_train_gpu.sh MODEL_NAME(pnet|rnet|onet) DEVICE_ID MINDRECORD_FILE"
echo "for example train PNet: bash run_standalone_train_gpu.sh pnet 0 pnet_train.mindrecord"
echo "=============================================================================================================="

MODEL=$1
DEVICE_ID=$2
MINDRECORD_FILE=$3

if [ $# -lt 3 ];
then
  echo "---------------------ERROR----------------------"
  echo "You must specify model name and gpu device and mindrecord file for training"
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

if [ $MODEL == "pnet" ];
then
  END_EPOCH=30
elif [ $MODEL == 'rnet' ];
then
  END_EPOCH=22
else
  END_EPOCH=22
fi

python $PROJECT_DIR/../train.py \
  --device_target GPU \
  --end_epoch $END_EPOCH \
  --model $MODEL \
  --mindrecord_file $MINDRECORD_FILE \
  --num_workers 8 > $LOG_DIR/training_gpu_$MODEL.log 2>&1 &

echo "The standalone train log is at /logs/training_gpu_$MODEL.log"
