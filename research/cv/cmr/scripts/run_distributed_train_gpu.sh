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
echo "bash run_distributed_train_gpu.sh DEVICE_NUM LOAD_CHECKPOINT_PATH"
echo "for example: bash run_distributed_train_gpu.sh 8 /path/load_ckpt"
echo "if no ckpt, just run: bash run_distributed_train_gpu.sh 8"
echo "=============================================================================================================="

DEVICE_NUM=$1
if [ $# == 2 ];
then
  LOAD_CHECKPOINT_PATH=$2
else
  LOAD_CHECKPOINT_PATH="None"
fi

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
LOG_DIR=$PROJECT_DIR/../logs
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

mpirun -n $DEVICE_NUM --allow-run-as-root python $PROJECT_DIR/../train.py \
  --distribute \
  --device_target GPU \
  --num_epochs 80 \
  --batch_size 16 \
  --keep_checkpoint_max 80 \
  --pretrained_checkpoint $LOAD_CHECKPOINT_PATH \
  --num_workers 2 \
  --save_checkpoint_dir 'checkpoints' \
  --checkpoint_steps 221 > ${LOG_DIR}/distribute_training_gpu.log 2>&1 &

echo "The distributed train log is at /logs/distribute_training_gpu.log"
