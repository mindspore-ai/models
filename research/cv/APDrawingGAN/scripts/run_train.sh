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


if [ $# != 9 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_train.sh [DATA_PATH] [LM_PATH] [BG_PATH] [CKPT_PATH] [AUXILIARY_PATH] [DEVICE_ID] [NITER] [SAVA_EPOCH_FREQ] [DEVICE_TARGET]"
    echo "for example: bash scripts/run_train.sh dataset/data/train dataset/landmark/ALL dataset/mask/ALL ckpt auxiliary/pretrain_APDGAN.ckpt 0 300 25 GPU"
    echo "=============================================================================================================="
exit 1
fi

DATA_PATH=$1
LM_PATH=$2
BG_PATH=$3
CKPT_PATH=$4
AUXILIARY_PATH=$5
DEVICE_ID=$6
NITER=$7
SAVA_EPOCH_FREQ=$8
DEVICE_TARGET=$9

ulimit -u unlimited
python train.py  \
  --device_id=$DEVICE_ID \
  --device_target=$DEVICE_TARGET \
  --dataroot=$DATA_PATH \
  --lm_dir=$LM_PATH\
  --bg_dir=$BG_PATH\
  --auxiliary_dir=$AUXILIARY_PATH\
  --ckpt_dir=$CKPT_PATH\
  --niter=$NITER  --save_epoch_freq=$SAVA_EPOCH_FREQ\
  --use_local --discriminator_local\
  --no_flip --no_dropout  --pretrain --isTrain > train.log 2>&1 &
