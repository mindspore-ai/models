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

if [ $# != 1 ] && [ $# != 2 ]
then
  echo "Usage: bash run_eval_gpu.sh [CKPT_FILE_OR_DIR] [VALIDATION_DATASET](optional)"
  exit 1
fi


if [ ! -d $1 ] && [ ! -f $1 ]
then
  echo "error: CKPT_FILE_OR_DIR=$1 is neither a directory nor a file"
  exit 1
fi

if [ $# == 2 ] && [ ! -d $2 ]
then
  echo "error: VALIDATION_DATASET=$2 is not a directory"
  exit 1
fi

ulimit -u unlimited

if [ $# == 1 ]
then
  GLOG_v=3 python eval.py \
    --checkpoint_path="$1" \
    --device_target="GPU" > "./eval.log" 2>&1 &
fi

if [ $# == 2 ]
then
  GLOG_v=3 python eval.py \
    --checkpoint_path="$1" \
    --val_data_path="$2" \
    --device_target="GPU" > "./eval.log" 2>&1 &
fi
