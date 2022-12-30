#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

if [ $# != 2 ]
then
    echo "Usage: bash run_eval_gpu.sh [DATASET] [DEVICE_ID]"
exit 1
fi

DATASET=$1
DEVICE_ID=$2

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export GLOG_v=3

echo "DATASET: $DATASET   DEVICE_ID: $DEVICE_ID"

if [ ! -e "checkpoints/$DATASET.ckpt" ]
then
    echo "ckpt file not exists"
exit 1
fi

if [ ! -d "logs" ]
then
    mkdir logs
fi

echo "Start evaluating..."
nohup python -u eval.py checkpoints/$DATASET.hpar checkpoints/$DATASET.ckpt --device GPU > logs/eval_$DATASET.log 2>&1 &