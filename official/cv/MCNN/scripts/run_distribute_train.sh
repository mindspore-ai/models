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

ulimit -u unlimited
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2
export RUN_OFFLINE=$3
export TRAIN_PATH=$4
export TRAIN_GT_PATH=$5
export VAL_PATH=$6
export VAL_GT_PATH=$7
export CKPT_PATH=$8

export SERVER_ID=0
rank_start=$((RANK_SIZE * SERVER_ID))

for((i=0; i<${RANK_SIZE}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py --rank_size=RANK_SIZE \
                    --run_offline=$RUN_OFFLINE --train_path=$TRAIN_PATH --train_gt_path=$TRAIN_GT_PATH \
                    --val_path=$VAL_PATH --val_gt_path=$VAL_GT_PATH --ckpt_path=$CKPT_PATH &> log &
    cd ..
done
