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
if [[ $# -ne 4 ]]; then
    echo "Usage: bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]"
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

LOGS_CKPT_DIR="$2"

if [ !  -d "$LOGS_CKPT_DIR" ]; then
  mkdir "$LOGS_CKPT_DIR"
  mkdir "$LOGS_CKPT_DIR/training_configs"
fi

DATASET_ROOT=$(get_real_path "$4")
CKPT_URL=$(get_real_path "$3")

cp ./*.py ./"$LOGS_CKPT_DIR"/training_configs
cp ./*.yaml ./"$LOGS_CKPT_DIR"/training_configs
cp -r ./cfg ./"$LOGS_CKPT_DIR"/training_configs
cp -r ./src ./"$LOGS_CKPT_DIR"/training_configs

python ./train.py \
    --device_target="GPU" \
    --device_id=0 \
    --logs_dir="$LOGS_CKPT_DIR" \
    --dataset_root="$DATASET_ROOT" \
    --ckpt_url="$CKPT_URL" \
    --lr=0.00125 \
    > ./"$2"/standalone_train.log 2>&1 &
