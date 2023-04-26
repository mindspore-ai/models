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
echo "bash run_distribute_train_gpu.sh DEVICE_NUM EPOCH_SIZE LR DATASET RANK_TABLE_FILE PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: bash run_distribute_train_gpu.sh 8 500 0.1 coco /data/hccl.json /opt/retinanet-500_458.ckpt(optional) 200(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 4 ] && [ $# != 6 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] \
[PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOG/train_log.txt"

export DEVICE_NUM=$1
EPOCH_SIZE=$2
LR=$3
DATASET=$4
PRE_TRAINED=$5
PRE_TRAINED_EPOCH_SIZE=$6

rm -rf LOG
mkdir ./LOG
cp ../*.py ./LOG
cp ../*.yaml ./LOG
cp -r ../src ./LOG
cd ./LOG || exit

echo "start training on GPU $DEVICE_NUM devices"
env > env.log

if [ $# == 4 ]
then
  mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout \
  python train.py  \
  --run_platform="GPU" \
  --batch_size=12 \
  --distribute=True  \
  --lr=$LR \
  --dataset=$DATASET \
  --device_num=$DEVICE_NUM  \
  --epoch_size=$EPOCH_SIZE > train_log.txt 2>&1 &
fi

if [ $# == 6 ]
then
  mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout \
  python train.py  \
  --run_platform="GPU" \
  --batch_size=12 \
  --distribute=True  \
  --lr=$LR \
  --dataset=$DATASET \
  --device_num=$DEVICE_NUM  \
  --pre_trained=$PRE_TRAINED \
  --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
  --epoch_size=$EPOCH_SIZE > train_log.txt 2>&1 &
fi
cd ..