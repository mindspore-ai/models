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

if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_distributed_train_gpu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME]"
    echo " or "
    echo "bash scripts/run_distributed_train_gpu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME] [PRETRAIN_CKPT]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

DATASET_ROOT=$(get_real_path "$1")
DATASET_NAME="$2"
MODEL_NAME="$3"
PRETRAIN_CKPT=$(get_real_path "$4")

CONFIG_FILE_BASE="./configs/${MODEL_NAME}_${DATASET_NAME}_8gpu_config.yaml"
CONFIG_FILE=$(get_real_path "$CONFIG_FILE_BASE")
DEV_NUM=8

# Check the pre-trained model checkpoint if it was specified
if [ $# == 4 ] && [ ! -f "$PRETRAIN_CKPT" ]
then
  echo "Cannot find the specified pre-trained model \"$PRETRAIN_CKPT\"."
  exit 1
fi

# Check the specified dataset root directory
if [ ! -d "$DATASET_ROOT" ]
then
  echo "The specified dataset root is not a directory: \"$DATASET_ROOT\"."
  exit 1
fi

# Check the configuration file (its name is derived from the model name and the dataset name)
if [ ! -f "$CONFIG_FILE" ]
then
  echo "Cannot find the configuration file \"$CONFIG_FILE_BASE\"."
  echo "The specified parameters DATASET_NAME=$DATASET_NAME and MODEL_NAME=$MODEL_NAME are incorrect or the configuration file is missing."
  exit 1
fi

# Specifying the log file
LOGS_DIR="train-logs"
LOG_FILE="./$LOGS_DIR/train_${MODEL_NAME}_${DATASET_NAME}_${DEV_NUM}gpu.log"

# Create a directory for logs if necessary
if [ ! -d "$LOGS_DIR" ]
then
  mkdir "$LOGS_DIR"
fi

# Run the training
if [ $# == 3 ]
then
  mpirun -np $DEV_NUM -merge-stderr-to-stdout --allow-run-as-root \
    python train.py \
      --config_path="$CONFIG_FILE" \
      --dataset_root="$DATASET_ROOT" \
      --is_train_distributed=True \
    > "$LOG_FILE" 2>&1 &
fi

if [ $# == 4 ]
then
  mpirun -np $DEV_NUM -merge-stderr-to-stdout --allow-run-as-root \
    python train.py \
      --config_path="$CONFIG_FILE" \
      --dataset_root="$DATASET_ROOT" \
      --is_train_distributed=True \
      --pre_trained="$PRETRAIN_CKPT" \
    > "$LOG_FILE" 2>&1 &
fi
