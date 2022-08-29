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

DATASET_NAME=$1
DATA_URL=$2
CKPT_URL=$3
EPOCH_NUM=$4
DEVICE_ID=$5

if [ $# -ne 5 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_standalone_train_gpu.sh DATASET_NAME DATA_URL CKPT_URL EPOCH_NUM DEVICE_ID"
  echo "For example: bash scripts/run_standalone_train_gpu.sh WIKI ./dataset ./ckpt 40 0"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
  exit 1
fi

python train.py  \
    --device_target "GPU" \
    --device_id $DEVICE_ID \
    --epochs $EPOCH_NUM \
    --dataset "$DATASET_NAME" \
    --data_url "$DATA_URL" \
    --ckpt_url "$CKPT_URL" \
    > train.log 2>&1 &
echo "start training"
cd ../
