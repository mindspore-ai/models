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

export DEVICE_ID=0
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ $# != 3 ]; then
    echo "Usage: \
bash run_eval_gpu.sh [SCALE] [DATASET_PATH] [CHECKPOINT_PATH]"
    exit 1
fi

if [ ! -f "$3" ]; then
    echo "error: CHECKPOINT_PATH:$3 does not exist"
    exit 1
fi

if [ ! -d "$2" ]; then
    echo "error: DATASET_PATH:$2 does not exist"
    exit 1
fi

if [ $1 -ne 2 ] && [ $1 -ne 4 ]
then
    echo "error: SCALE=$1 is not 2 or 4."
exit 1
fi

nohup python ../eval.py \
    --scale $1 \
    --dataset_GT_path $2 \
    --device_target GPU \
    --resume_state $3 \
    > ../eval.log 2>&1 &
pid=$!
echo "Start evaluating with rank ${RANK_ID} on device ${DEVICE_ID}: ${pid}"


