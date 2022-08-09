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

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/run_eval_ascend.sh 
    [PRETRAIN_CKPT] [DEVICE_ID]"
exit 1
fi

log_dir="./eval_out"

if [ ! -d $log_dir ]; then
    mkdir eval_out
fi

DEVICE_ID=$1
PRETRAIN_CKPT=$2

DEVICE_ID=$DEVICE_ID python eval.py --device_target 'Ascend' --pretrain_ckpt $PRETRAIN_CKPT > eval_out/eval.log 2>&1 &
