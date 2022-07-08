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

if [ $# != 4 ]
then
    echo "Usage: bash script/run_eval_gpu.sh [CKPT_PATH] [DEVICE_ID] [MEASURE_MSSIM] [DATASET_DIR]"
    exit 1
fi

export CKPT=$1
export DEVICE_ID=$2
export MEASURE_MSSIM=$3
export DATASET_DIR=$4

python -u eval.py \
       --checkpoint_g="$CKPT" \
       --device_target GPU \
       --device_id="$DEVICE_ID" \
       --measure_ms_ssim="$MEASURE_MSSIM" \
       --original_img_dir="$DATASET_DIR" \
       > eval.log 2>&1 &
