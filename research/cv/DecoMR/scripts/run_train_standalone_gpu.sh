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

if [ $# != 6 ]
then
  echo "==============================================================================================================="
  echo "Please run the script as: "
  echo "bash ./run_train_standalone_gpu.sh [DATASET] [DEVICE_ID] [EPOCHS_DP] [EPOCHS_END] [BATCH_SIZE] [CKPT_PATH]"
  echo "for example: bash ./run_train_standalone_gpu.sh up-3d 0 5 30 16 './ckpt'"
  echo "==============================================================================================================="
  exit 1
fi

ulimit -u unlimited

DATASET=$1
DEVICE_ID=$2
EPOCHS_DP=$3
EPOCHS_END=$4
BATCH_SIZE=$5
CKPT_PATH=$6

cd ..

python train.py  \
  --dataset=$DATASET \
  --device_id=$DEVICE_ID \
  --num_epochs_dp=$EPOCHS_DP \
  --num_epochs_end=$EPOCHS_END \
  --batch_size=$BATCH_SIZE \
  --ckpt_dir=$CKPT_PATH > trainx1.log 2>&1 &
