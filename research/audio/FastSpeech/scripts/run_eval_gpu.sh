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
    echo "Usage: bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET_PATH] [FS_CKPT_URL] [WG_CKPT_URL] [DS_CKPT_URL]"
exit 1;
fi

export CUDA_VISIBLE_DEVICES=$1

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

CONFIG_FILE_BASE="./default_config.yaml"
OUTPUT_DIR_BASE="./results"
OUTPUT_ROOT=$(get_real_path "$OUTPUT_DIR_BASE")
CONFIG_FILE=$(get_real_path "$CONFIG_FILE_BASE")
DATASET_ROOT=$(get_real_path "$2")
FS_CKPT=$(get_real_path "$3")
WG_CKPT=$(get_real_path "$4")
DS_CKPT=$(get_real_path "$5")

python eval.py \
    --device_target="GPU" \
    --device_id=0 \
    --output_dir="$OUTPUT_ROOT" \
    --dataset_path="$DATASET_ROOT" \
    --config_path="$CONFIG_FILE" \
    --fs_ckpt_url="$FS_CKPT" \
    --wg_ckpt_url="$WG_CKPT" \
    --ds_ckpt_url="$DS_CKPT" \
    > eval.log 2>&1 &
