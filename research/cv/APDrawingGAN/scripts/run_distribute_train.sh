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

if [ $# != 8 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [LM_PATH] [BG_PATH] [CKPT_PATH] [AUXILIARY_PATH] [EPOCH] [SAVA_EPOCH_FREQ]"
    echo "for example: bash scripts/run_distribute_train.sh hccl_8p_01_127.0.0.1.json dataset/data/train dataset/landmark/ALL dataset/mask/ALL ckpt auxiliary/pretrain_APDGAN.ckpt 300 25"
    echo "=============================================================================================================="
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
RANK_TABLE_FILE=$(realpath $1)
export RANK_TABLE_FILE
DATA_PATH=$2
LM_PATH=$3
BG_PATH=$4
CKPT_PATH=$5
AUXILIARY_PATH=$6
NITER=$7
SAVA_EPOCH_FREQ=$8
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cp ./train.py ./train_parallel$i
    cp ./config_train.yaml ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./train_parallel$i ||exit
    env > env.log
    python train.py \
        --run_distribute True\
        --batch_size 2 \
        --dataroot=../$DATA_PATH \
        --lm_dir=../$LM_PATH \
        --bg_dir=../$BG_PATH \
        --auxiliary_dir=../$AUXILIARY_PATH \
        --ckpt_dir=$CKPT_PATH \
        --niter $NITER  --save_epoch_freq $SAVA_EPOCH_FREQ \
        --use_local --discriminator_local \
        --no_flip --no_dropout --pretrain --isTrain > train.log 2>&1 &
    cd ..
done
