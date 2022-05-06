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
if [[ $# -ne 3 ]]; then
    echo "Usage: bash ./scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [DATASET_ROOT]"
exit 1;
fi

export RANK_SIZE=$1

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

CONFIG_FILE_BASE="./default_config.yaml"
CONFIG_FILE=$(get_real_path "$CONFIG_FILE_BASE")
DATASET_ROOT=$(get_real_path "$3")
LOGS_ROOT=$(get_real_path "$2")

if [ !  -d "$LOGS_ROOT" ]; then
  mkdir "$LOGS_ROOT"
  mkdir "$LOGS_ROOT/training_configs"
fi

cp ./*.py "$LOGS_ROOT"/training_configs
cp ./*.yaml "$LOGS_ROOT"/training_configs
cp -r ./src "$LOGS_ROOT"/training_configs

mpirun -n $1 --allow-run-as-root \
    python train.py  \
    --device_target="GPU" \
    --logs_dir="$LOGS_ROOT" \
    --dataset_path="$DATASET_ROOT" \
    --config_path="$CONFIG_FILE" \
    --epochs=300 \
    --lr_scale=2 \
    > "$LOGS_ROOT"/distribute_train.log 2>&1 &
