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

if [ $# != 3 ]
then
    echo "Usage: bash scripts/run_export_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH]"
exit 1
fi

DEVICE_ID=$1
CFG_PATH=$2
CKPT_PATH=$3
SAVE_PATH="$(dirname "$CKPT_PATH")"

export DEVICE_ID="$DEVICE_ID"

python export.py \
  --device_target=GPU \
  --config_path="$CFG_PATH" \
  --ckpt_path="$CKPT_PATH" \
  --aux_loss=0 > "$SAVE_PATH"/log_export.txt 2>&1 &
