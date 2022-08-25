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
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_train.sh [CHECKPOINT_ID] [DATA_NAME] [CUDA_VISIBLE_DEVICES]/[DEVICE_ID] [DEVICE_TARGET]
    CHECKPOINT_ID means model checkpoint id.
    DATA_NAME means dataset name, it's value is 'steam'.
    CUDA_VISIBLE_DEVICES means cuda visible device id.
    DEVICE_ID means device id, it can be set by environment variable DEVICE_ID.
    DEVICE_TARGET is optional, it's value is ['GPU', 'Ascend'], default 'GPU'."
exit 1
fi

CHECKPOINT_ID=$1
DATA_NAME=$2

DEVICE_TARGET='GPU'
if [ $# == 4 ]; then
    DEVICE_TARGET=$4
fi

if [ "$DEVICE_TARGET" = "GPU" ];
then
  export CUDA_VISIBLE_DEVICES=$3
  python eval.py --checkpoint_id $CHECKPOINT_ID --dataset $DATA_NAME --device_target $DEVICE_TARGET \
       --device_id 0  &> scripts/eval_standalone_gpu_log &
fi

if [ "$DEVICE_TARGET" = "Ascend" ];
then
  export DEVICE_ID=$3
  python eval.py --checkpoint_id $CHECKPOINT_ID --dataset $DATA_NAME --device_target $DEVICE_TARGET \
       --device_id $DEVICE_ID  &> scripts/eval_standalone_gpu_log &
fi