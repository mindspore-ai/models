#! /bin/bash
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

if [ $# != 3 ]
then
  echo "Usage: bash scripts/run_distribute_train_gpu.sh DATASET_PATH DEVICE_NUM DEVICE_ID"
  echo "Example: bash scripts/run_distribute_train_gpu.sh /home/name/cityscapes 4 0,1,2,3"
  exit 1
fi

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
DATASET_PATH=$1
export RANK_SIZE=$2
export CUDA_VISIBLE_DEVICES=$3

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

cd $BASE_PATH/../

if [ -d "distribute_train_gpu" ]
then
    echo "delete old logs!"
    rm -rf distribute_train_gpu
fi

mkdir distribute_train_gpu
cd distribute_train_gpu

echo "Start train stage 1"
mpirun -n $RANK_SIZE --allow-run-as-root \
python -u $BASE_PATH/../train.py \
    --lr=1e-3 \
    --repeat=2 \
    --run_distribute=true \
    --device_target='GPU' \
    --save_path="./checkpoint/" \
    --mindrecord_train_data="$BASE_PATH/../data/train.mindrecord" \
    --stage=1 \
    --ckpt_path="" \
    > log_stage1.txt 2>&1
wait

echo "Start train stage 2"
mpirun -n $RANK_SIZE --allow-run-as-root \
python -u $BASE_PATH/../train.py \
    --lr=1e-3 \
    --repeat=2 \
    --run_distribute=true \
    --device_target='GPU' \
    --save_path="./checkpoint/" \
    --mindrecord_train_data="$BASE_PATH/../data/train.mindrecord" \
    --stage=2 \
    --ckpt_path="./checkpoint/Encoder_stage1.ckpt" \
    > log_stage2.txt 2>&1
wait

echo "Start train stage 3"
mpirun -n $RANK_SIZE --allow-run-as-root \
python -u $BASE_PATH/../train.py \
    --lr=1e-3 \
    --repeat=2 \
    --run_distribute=true \
    --device_target='GPU' \
    --save_path='./checkpoint/' \
    --mindrecord_train_data="$BASE_PATH/../data/train.mindrecord" \
    --stage=3 \
    --ckpt_path="./checkpoint/Encoder_stage2.ckpt" \
    > log_stage3.txt 2>&1
wait

echo "Start train stage 4"
mpirun -n $RANK_SIZE --allow-run-as-root \
python -u $BASE_PATH/../train.py \
    --lr=1e-3 \
    --repeat=2 \
    --run_distribute=true \
    --device_target='GPU' \
    --save_path='./checkpoint/' \
    --mindrecord_train_data="$BASE_PATH/../data/train.mindrecord" \
    --stage=4 \
    --ckpt_path="./checkpoint/ERFNet_stage3.ckpt" \
    > log_stage4.txt 2>&1

