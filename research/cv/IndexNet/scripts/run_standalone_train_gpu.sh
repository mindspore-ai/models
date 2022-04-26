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
if [[ $# -ne 5 ]]; then
    echo "Usage: bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [MOBILENET_CKPT] [DATA_DIR] [BG_DIR]"
exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

LOGS_DIR=$(get_real_path "$2")
CKPT_URL=$(get_real_path "$3")
DATA_DIR=$(get_real_path "$4")
BG_DIR=$(get_real_path "$5")

if [ !  -d "$LOGS_DIR" ]; then
  mkdir "$LOGS_DIR"
  mkdir "$LOGS_DIR/training_configs"
fi

cp ./*.py "$LOGS_DIR"/training_configs
cp ./*.yaml "$LOGS_DIR"/training_configs
cp -r ./src "$LOGS_DIR"/training_configs

python train.py \
    --device_target="GPU" \
    --device_id=0 \
    --logs_dir="$LOGS_DIR" \
    --ckpt_url="$CKPT_URL" \
    --data_dir="$DATA_DIR" \
    --bg_dir="$BG_DIR" \
    > "$LOGS_DIR"/standalone_train.log 2>&1 &
