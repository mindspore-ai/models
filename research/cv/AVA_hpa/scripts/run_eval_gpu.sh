#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
echo "bash run_eval_gpu.sh [MODEL_ARCH] [CLASS_NUM] [CKPT_PATH] [DATA_DIR] [SAVE_EVAL_PATH]"
echo "=============================================================================================================="

PROJECT_DIR=$(
  cd "$(dirname "$0")" || exit
  pwd
)

if [ ! -f $3 ]
then
  echo "error: CKPT_PATH=$3 is not a file"
  exit 1
fi

if [ ! -d $4 ]
then
  echo "error: DATA_DIR=$4 is not a directory"
  exit 1
fi

if [ ! -d $5 ]
then
  echo "error: SAVE_EVAL_PATH=$5 is not a directory"
  exit 1
fi

python ${PROJECT_DIR}/../eval.py \
  --device_target=GPU \
  --model_arch=$1 \
  --classes=$2 \
  --ckpt_path=$3 \
  --data_dir=$4 \
  --save_eval_path=$5 &> log &