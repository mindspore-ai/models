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

if [ $# != 3 ] && [ $# != 2 ]
then
    echo "===================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [SAVE_IMG_DIR](optional)"
    echo "for example:"
    echo "bash scripts/run_eval_gpu.sh /path/to/dataset/root /path/to/eval/checkpoint.ckpt /path/to/result/images"
    echo "or"
    echo "bash scripts/run_eval_gpu.sh /path/to/dataset/root /path/to/eval/checkpoint.ckpt"
    echo "===================================================================================================="
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD/$1"
  fi
}

DATASET_PATH=$(get_real_path "$1")
CKPT_PATH=$(get_real_path "$2")

if [ $# == 3 ]
then
    SAVE_IMG_DIR=$(get_real_path "$3")
else
    SAVE_IMG_DIR=""
fi

if [ ! -d "$DATASET_PATH" ] ; then
    echo "Cannot find the specified dataset directory: $DATASET_PATH"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ] ; then
    echo "Cannot find the specified checkpoint: $CKPT_PATH"
    exit 1
fi

if [ -d eval_logs ]
then
  rm -r eval_logs
fi

mkdir eval_logs

python eval.py --dataset_root "$DATASET_PATH" \
               --ckpt_file "$CKPT_PATH" \
               --img_save_directory "$SAVE_IMG_DIR" \
               > ./eval_logs/eval.log 2>&1 &
