#! /bin/bash
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

if [ $# != 3 ]
then
  echo "Usage: bash scripts/run_standalone_train.sh DEVICE_TARTGET DATASET_PATH DEVICE_ID"
  echo "Example: bash scripts/run_standalone_train.sh [Ascend/GPU] /home/name/cityscapes 0"
  exit 1
fi

echo "DEVICE_TARTGET: $1"
echo "CITYSCAPES DATASET PATH: $2"
echo "DEVICE_ID: $3"

DEVICE_TARTGET=$1
DATASET_PATH=$2
export RANK_SIZE=1
export DEVICE_ID=$3
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$1 is not a directory"
exit 1
fi

cd $BASE_PATH/../

if [ -d "standalone_train" ]
then
    echo "delete old logs!"
    rm -rf standalone_train
fi

mkdir standalone_train
cd standalone_train

python -u ../train.py \
    --device_target=$DEVICE_TARTGET \
    --lr=5e-4 \
    --repeat=1 \
    --run_distribute=false \
    --save_path='./checkpoint' \
    --mindrecord_train_data="../data/train.mindrecord" \
    --stage=1 \
    --ckpt_path="" \
    > log_stage1.txt 2>&1

python -u ../train.py \
    --device_target=$DEVICE_TARTGET \
    --lr=5e-4 \
    --repeat=1 \
    --run_distribute=false \
    --save_path='./checkpoint' \
    --mindrecord_train_data="../data/train.mindrecord" \
    --stage=2 \
    --ckpt_path="./checkpoint/Encoder_stage1.ckpt" \
    > log_stage2.txt 2>&1

python -u ../train.py \
    --device_target=$DEVICE_TARTGET \
    --lr=5e-4 \
    --repeat=1 \
    --run_distribute=false \
    --save_path='./checkpoint' \
    --mindrecord_train_data="../data/train.mindrecord" \
    --stage=3 \
    --ckpt_path="./checkpoint/Encoder_stage2.ckpt" \
    > log_stage3.txt 2>&1

python -u ../train.py \
    --device_target=$DEVICE_TARTGET \
    --lr=5e-4 \
    --repeat=1 \
    --run_distribute=false \
    --save_path='./checkpoint' \
    --mindrecord_train_data="../data/train.mindrecord" \
    --stage=4 \
    --ckpt_path="./checkpoint/ERFNet_stage3.ckpt" \
    > log_stage4.txt 2>&1

python -u ../eval.py \
    --device_target=$DEVICE_TARTGET \
    --data_path=$DATASET_PATH \
    --run_distribute=false \
    --encode=false \
    --model_root_path="./checkpoint" \
    --device_id=$DEVICE_ID \
    > log_eval.txt 2>&1 &

