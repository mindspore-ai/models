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
# Parameters!
if [ $# != 2 ]
then
    echo "Usage: bash run_standalone_train.sh [RANK_TABLE_FILE] [DEVICE_NUM]."
    exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DEVICE_NUM]."
exit 1
fi

nowtime=$(date '+%Y%m%d' )
if [ -d 'scripts/'$nowtime ];
then
    rm -rf 'scripts/'$nowtime
fi

mkdir 'scripts/'$nowtime

train_path="data/ucf101_rgb_train_split_1.txt"
val_path="data/ucf101_rgb_val_split_1.txt"
#############################################
#--- training hyperparams ---
dataset_name="ucf101"
netType="ECO"
batch_size=16
learning_rate=0.008
num_segments=4
dropout=0.7
resume='ms_model_kinetics_checkpoint0720.ckpt'
#####################################################################
export RANK_TABLE_FILE=$1
export DEVICE_NUM=$2
export RANK_SIZE=$2
export HCCL_CONNECT_TIMEOUT=360
for((i=0;i<${RANK_SIZE};i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=${i}
    echo "start distributed training for rank $RANK_ID, device $DEVICE_ID"
    python3 -u train.py --dataset ${dataset_name} --modality RGB --train_list ${train_path} --val_list ${val_path} --arch ${netType} --num_segments ${num_segments}  --lr ${learning_rate} --num_saturate 5 --epochs 60 --batch-size ${batch_size} --dropout ${dropout} --consensus_type identity --rgb_prefix img_  --no_partialbn True --nesterov True --run_distribute=True --resume ${resume} --device_id ${DEVICE_ID} &>./scripts/$nowtime/train$i.log &
done
