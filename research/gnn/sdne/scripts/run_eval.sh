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

DATASET_NAME=$1
DATA_URL=$2
CHECKPOINT=$3
DEVICE_ID=$4

if [ $# -ne 4 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_eval.sh DATASET_NAME DATA_URL CHECKPOINT DEVICE_ID"
  echo "For example: bash run_eval.sh NAME /path/dataset /path/ckpt device_id"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
  exit 1
fi

python eval.py  \
    --device_id $DEVICE_ID \
    --checkpoint "$CHECKPOINT" \
    --data_url "$DATA_URL" \
    --dataset "$DATASET_NAME" \
    > eval.log 2>&1 &
echo "start evaluation"
exit 0
