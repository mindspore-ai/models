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

if [ $# -ne 3 ]; then
  echo "Usage: bash scripts/run_onnx_eval_gpu.sh [DATASET_PATH] [DEVICE_ID] [PRETRAINED_MODEL]"
  exit 1
fi

run_gpu() {
  ulimit -u unlimited
  echo "start evaluation for device $2"
  env >env.log
  python eval_onnx.py --test_mindrecord=$1 --device_id=$2 --pretrained_model=$3 > log_eval_onnx.txt 2>&1 &
  cd ..
}

run_gpu $1 $2 $3
