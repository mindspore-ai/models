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
if [ $# != 8 ]; then
  echo "Usage: bash ./scripts/run_joint_eval.sh [device_id] [dataset] [video_path] [video_path_joint_flow] [annotation_path] [annotation_path_joint_flow] [rgb_path] [flow_path]"
  exit 1
fi

nohup python ./eval.py \
  --device_id=$1 \
  --dataset=$2 \
  --video_path=$3 \
  --video_path_joint_flow=$4 \
  --annotation_path=$5 \
  --annotation_path_joint_flow=$6 \
  --rgb_path=$7 \
  --flow_path=$8 \
  --test_mode=joint >./joint_eval_log.txt 2>&1 &
