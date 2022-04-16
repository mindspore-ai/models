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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_train_distribute_GPU.sh [DATA_PATH] [LM_PATH] [BG_PATH] [CKPT_PATH] [AUXILIARY_PATH] [NITER] [SAVA_EPOCH_FREQ]"
echo "for example: bash scripts/run_train_distribute_GPU.sh /dataset/data/train /dataset/landmark/ALL dataset/train/mask/ALL ckpt /auxiliary/pretrain_APDGAN.ckpt 300 25"
echo "=============================================================================================================="

DATA_PATH=$1
LM_PATH=$2
BG_PATH=$3
CKPT_PATH=$4
AUXILIARY_PATH=$5
NITER=$6
SAVA_EPOCH_FREQ=$7

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

mpirun --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout -np 8 python train.py \
  --run_distribute True\
  --device_target GPU \
  --batch_size 8 \
  --dataroot=$DATA_PATH \
  --lm_dir=$LM_PATH \
  --bg_dir=$BG_PATH \
  --auxiliary_dir=$AUXILIARY_PATH \
  --ckpt_dir=$CKPT_PATH \
  --niter $NITER  --save_epoch_freq $SAVA_EPOCH_FREQ \
  --use_local --discriminator_local \
  --no_flip --no_dropout --pretrain --isTrain > output.train.log 2>&1 &
