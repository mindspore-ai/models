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
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_train.sh [DATA_NAME] [CUDA_VISIBLE_DEVICES]/[DEVICE_ID] [DEVICE_TARGET]
    DATA_NAME means dataset name, it's value is 'steam'.
    CUDA_VISIBLE_DEVICES means cuda visible device id.
    DEVICE_ID means device id, it can be set by environment variable DEVICE_ID.
    DEVICE_TARGET is optional, it's value is ['GPU', 'Ascend'], default 'GPU'."
exit 1
fi

DATA_NAME=$1
DEVICE_TARGET='GPU'

if [ $# == 3 ]; then
    DEVICE_TARGET=$3
fi


if [ "$DEVICE_TARGET" = "GPU" ];
then
  export CUDA_VISIBLE_DEVICES=$2
  python preprocess_dataset.py --dataset $DATA_NAME --device_target $DEVICE_TARGET &> scripts/train_standalone_log &&
  python train.py --dataset $DATA_NAME --device_target $DEVICE_TARGET --device_id 0 &>> scripts/train_standalone_log &
fi

if [ "$DEVICE_TARGET" = "Ascend" ];
then
  export DEVICE_ID=$2
  python preprocess_dataset.py --dataset $DATA_NAME --device_target $DEVICE_TARGET &> scripts/train_standalone_log &&
  python train.py --dataset $DATA_NAME --device_target $DEVICE_TARGET --device_id $DEVICE_ID &>> scripts/train_standalone_log &
fi