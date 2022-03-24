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

if [ $# != 2 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_distributed_train_gpu.sh [DATASET_ROOT] [PRETRAIN_BNINCEPTION_CKPT]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

DEV_NUM=8
DATASET_ROOT=$(get_real_path "$1")
PRETRAIN_BNINCEPTION_CKPT=$(get_real_path "$2")

# Check the pre-trained model checkpoint
if [ ! -f "$PRETRAIN_BNINCEPTION_CKPT" ]
then
  echo "Cannot find the specified pre-trained BNInception model \"$PRETRAIN_BNINCEPTION_CKPT\"."
  exit 1
fi

# Check the specified dataset root directory
if [ ! -d "$DATASET_ROOT" ]
then
  echo "The specified dataset root is not an existing directory: \"$DATASET_ROOT\"."
  exit 1
fi

# Specifying the log file
LOGS_DIR="train-logs"
LOG_FILE="./$LOGS_DIR/train_${DEV_NUM}gpu.log"

# Create a directory for logs if necessary
if [ ! -d "$LOGS_DIR" ]
then
  mkdir "$LOGS_DIR"
fi

# Run the training
echo "Start distributed training in the background."

mpirun -np $DEV_NUM -merge-stderr-to-stdout --allow-run-as-root \
  python train.py --dataset_root="$DATASET_ROOT" \
    --is_train_distributed=True \
    --train_workers=4 \
    --train_batch_size=8 \
    --pre_trained_backbone="$PRETRAIN_BNINCEPTION_CKPT" > "$LOG_FILE" 2>&1 &
