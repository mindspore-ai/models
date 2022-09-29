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

cd ..

# help message
if [ $# != 4 ]; then
  echo "cd scripts"
  echo "Usage: bash run_eval.sh [device_id] [context] [ckpt_path] [is_fp16]"
  exit 1
fi

if [ ! -d "./eval" ];then
  mkdir ./eval
fi

echo 'dir created...'


nohup python ./eval.py \
  --device_id=$1 \
  --context=$2 \
  --ckpt_path=$3 \
  --is_fp16=$4>./eval/eval_$2_log.txt 2>&1 &

echo 'start evaluation...'
