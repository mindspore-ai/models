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
    echo "Usage: bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [CKPT_URL] [DATASET_ROOT]"
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

CKPT_URL=$(get_real_path "$2")
DATASET_ROOT=$(get_real_path "$3")

if [ ! -d "$DATASET_ROOT" ]; then
    echo "The specified dataset root is not a directory: $DATASET_ROOT"
    exit 1;
fi

if [ ! -f "$CKPT_URL" ]; then
    echo "The specified checkpoint does not exist: $CKPT_URL"
    exit 1;
fi

python ./eval.py \
    --device_target="GPU" \
    --device_id=0 \
    --ckpt_url="$CKPT_URL" \
    --dataset_root="$DATASET_ROOT" \
    > ./eval.log 2>&1 &
