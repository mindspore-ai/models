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
    echo "bash scripts/run_eval_gpu.sh [DATASET_ROOT] [CKPT_PATH]"
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
CKPT_PATH=$(get_real_path "$2")

# Check the specified checkpoint path
if [ ! -f "$CKPT_PATH" ]
then
  echo "Cannot find the specified model checkpoint \"$CKPT_PATH\"."
  exit 1
fi

# Check the specified dataset root directory
if [ ! -d "$DATASET_ROOT" ]
then
  echo "The specified dataset root is not an existing directory: \"$DATASET_ROOT\"."
  exit 1
fi

# Specifying the log file
LOGS_DIR="eval-logs"
LOG_FILE="./$LOGS_DIR/eval.log"

# Create a directory for logs if necessary
if [ ! -d "$LOGS_DIR" ]
then
  mkdir "$LOGS_DIR"
fi

# Run evaluation
echo "Start evaluation in the background."
python eval.py  --dataset_root="$DATASET_ROOT" --ckpt_file="$CKPT_PATH" > "$LOG_FILE" 2>&1 &
