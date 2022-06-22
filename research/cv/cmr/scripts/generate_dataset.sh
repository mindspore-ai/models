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
echo "Before run the script, you must ensure that you have configure the datasets file path in src/config.py correctly"
echo "If you want to only generate datasets for training, please run the script as: "
echo "bash preprocess_dataset.sh train"
echo "Or if you want to only generate datasets for evaluating, please run the script as: "
echo "bash preprocess_dataset.sh eval"
echo "Both generate datasets for training and evaluating, please run the script as:"
echo "bash preprocess_dataset.sh train eval"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
LOG_DIR=$PROJECT_DIR/../logs

if [ $# == 2 ]
then
  python $PROJECT_DIR/../preprocess.py --train_files --eval_files > $LOG_DIR/generate_train_eval_files.log 2>&1 &
elif [ $# == 1 ]
then
  if [ $1 == "train" ]
  then
    python $PROJECT_DIR/../preprocess.py --train_files > $LOG_DIR/generate_train_files.log 2>&1 &
  else
    python $PROJECT_DIR/../preprocess.py --eval_files  > $LOG_DIR/generate_eval_files.log 2>&1 &
  fi
fi

echo "The train dataset generation log is at /logs/generate_train_files.log and eval dataset generation
log is at generate_eval_files.log"
