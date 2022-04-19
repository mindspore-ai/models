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

if [[ $# != 3 ]]; then
    echo "Usage: bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [EDGE_PATH] [CKPT_NAME]"
    exit 1
fi

EDGE_PATH=$2
CKPT_NAME=$3

if [ ! -d "logs" ]; then
    mkdir logs
fi

export DEVICE_ID=$1

nohup python -u train.py \
  --device_target="GPU" \
  --device_id="$1" \
  --edge-path="$EDGE_PATH" \
  --features-path="$EDGE_PATH" \
  --checkpoint_file="./logs/standalone_$CKPT_NAME" > logs/standalone_train.log 2>&1 &
