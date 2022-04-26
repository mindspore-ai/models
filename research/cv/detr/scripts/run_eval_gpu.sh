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

if [ $# != 5 ]
then
    echo "Usage: bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [CKPT_PATH] [DATASET_PATH]"
exit 1
fi

DEVICE_ID=$1
CFG_PATH=$2
SAVE_PATH=$3
CKPT_PATH=$4
DATASET_PATH=$5

if [ ! -d "$SAVE_PATH" ];
then
  mkdir "$SAVE_PATH"
fi

export DEVICE_ID="$DEVICE_ID"

python eval.py \
  --device_target=GPU \
  --config_path="$CFG_PATH" \
  --ckpt_path="$CKPT_PATH" \
  --eval=1 \
  --aux_loss=0 \
  --coco_path="$DATASET_PATH" \
  --batch_size=1 > "$SAVE_PATH"/log_eval.txt 2>&1 &
