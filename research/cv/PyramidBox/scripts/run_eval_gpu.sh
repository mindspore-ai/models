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
echo "bash run_eval_gpu.sh PYRAMIDBOX_CKPT"
echo "for example: bash run_eval_gpu.sh pyramidbox.ckpt"
echo "=============================================================================================================="

if [ $# -lt 1 ];
then
  echo "---------------------ERROR----------------------"
  echo "You must specify pyramidbox checkpoint"
  exit
fi

CKPT=$1

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
LOG_DIR=$PROJECT_DIR/../logs
if [ ! -d $LOG_DIR ]
then
  mkdir $LOG_DIR
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python $PROJECT_DIR/../eval.py --model $CKPT > $LOG_DIR/eval_gpu.log 2>&1 &
