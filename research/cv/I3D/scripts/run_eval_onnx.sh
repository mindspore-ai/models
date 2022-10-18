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



if [ $1 == "joint" ]; then
  echo "joint mode"
  if [ $# -ne 10 ]; then
    echo "Usage: bash ./scripts/run_eval_onnx.sh [test_mode] [dataset] [video_path] [video_path_joint_flow] [annotation_path] [annotation_path_joint_flow] [rgb_path] [flow_path]"
    exit 1
  fi
  nohup python ./eval_onnx.py \
    --device_target=$2 \
    --device_id=$3 \
    --dataset=$4 \
    --video_path=$5 \
    --video_path_joint_flow=$6 \
    --annotation_path=$7 \
    --annotation_path_joint_flow=$8 \
    --rgb_path=$9 \
    --flow_path=${10} \
    --test_mode=joint >./joint_onnx_eval.log 2>&1 &
elif [ $1 == "rgb" ]; then
  echo "single mode"
  if [ $# -ne 7 ]; then
    echo "Usage: bash ./scripts/run_eval_onnx.sh [test_mode] [dataset] [video_path] [annotation_path] [rgb_path]"
    exit 1
  fi
  nohup python ./eval_onnx.py \
    --device_target=$2 \
    --device_id=$3 \
    --dataset=$4 \
    --video_path=$5 \
    --annotation_path=$6 \
    --rgb_path=$7 \
    --test_mode=rgb >./rgb_onnx_eval.log 2>&1 & 
elif [ $1 == "flow" ]; then
  echo "single mode"
  if [ $# -ne 7 ]; then
    echo "Usage: bash ./scripts/run_eval_onnx.sh [test_mode] [dataset] [video_path] [annotation_path] [flow_path]"
    exit 1
  fi
  nohup python ./eval_onnx.py \
    --device_target=$2 \
    --device_id=$3 \
    --dataset=$4 \
    --video_path=$5 \
    --annotation_path=$6 \
    --flow_path=$7 \
    --test_mode=flow >./flow_onnx_eval.log 2>&1 &
else
  echo "illegal eval mode"
fi
