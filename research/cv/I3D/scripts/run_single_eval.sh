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

# help message
if [ $# != 6 ]; then
  echo "Usage: bash ./scripts/run_single_eval.sh [device_id] [mode] [dataset] [ckpt_path] [video_path] [annotation_path]"
  exit 1
fi

nohup python ./eval.py \
  --device_id=$1 \
  --test_mode=$2 \
  --$2_path=$4 \
  --dataset=$3 \
  --video_path=$5 \
  --annotation_path=$6 >./single_eval_$2_log.txt 2>&1 &
