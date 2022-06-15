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
if [ $# != 4 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_tacred_eval.sh [DATA] [CHECKPOINT_FILE] [MODEL_FILE] [EVAL_BATCH_SIZE]"
    exit 1
fi


get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}
DATA=$(get_real_path "$1")
CHECKPOINT_FILE=$(get_real_path "$2")
MODEL_FILE=$(get_real_path "$3")
EVAL_BATCH_SIZE=$4

if [ ! -d "$MODEL_FILE" ]
then
  echo "The specified model file  root is not a directory: \"$MODEL_FILE\"."
  echo "[MODEL_FILE] is the path to the folder that contains the unpacked luke_large_500k.tar.gz files."
  exit 1
fi

if [ ! -d "$DATA" ]
then
  echo "The specified dataset root is not a directory: \"$DATA\"."
  exit 1
fi

if [ ! -f "$CHECKPOINT_FILE" ] && [ ! -d "$CHECKPOINT_FILE" ]
then
  echo "The specified path is not a checkpoint file or checkpoint directory: \"$CHECKPOINT_FILE\"."
  exit 1
fi

if [ ! -d "logs" ]
then
  mkdir logs
fi

python run_tacred_eval.py --data="$DATA" --checkpoint_file="$CHECKPOINT_FILE" --model_file="$MODEL_FILE" --eval_batch_size="$EVAL_BATCH_SIZE" \
        > ./logs/eval.log 2>&1 &
