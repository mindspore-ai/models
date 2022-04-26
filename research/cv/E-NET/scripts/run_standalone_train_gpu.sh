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

if [ $# != 2 ]
then
  echo "Usage: bash scripts/run_distribute_train_gpu.sh  DEVICE_ID /path/to/cityscapes"
  echo "Example: bash scripts/run_distribute_train_gpu.sh 4 /home/name/cityscapes"
  exit 1
fi

if [ ! -d $2 ]
then
    echo "error: DATASET_PATH=$2 is not a directory"
exit 1
fi

# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)


mkdir ./log_single_device
cd ./log_single_device
mkdir ./checkpoint

echo "DEVICE_ID: $1"
echo "cityscapes_path: $2"
export DEVICE_ID=$1
export RANK_SIZE=1
cityscapes_path=$2

python -u $BASE_PATH/../train.py \
    --lr 5e-4 \
    --repeat 1 \
    --run_distribute false \
    --save_path './checkpoint' \
    --mindrecord_train_data "../data/train.mindrecord" \
    --stage 1 \
    --ckpt_path "" \
    --device_target GPU \
    > log_stage1.txt 2>&1

python -u $BASE_PATH/../train.py \
    --lr 5e-4 \
    --repeat 1 \
    --run_distribute false \
    --save_path './checkpoint' \
    --mindrecord_train_data "../data/train.mindrecord" \
    --stage 2 \
    --ckpt_path "./checkpoint/Encoder_stage1.ckpt" \
    --device_target GPU \
    > log_stage2.txt 2>&1

python -u $BASE_PATH/../train.py \
    --lr 5e-4 \
    --repeat 1 \
    --run_distribute false \
    --save_path './' \
    --mindrecord_train_data "../data/train.mindrecord" \
    --stage 3 \
    --ckpt_path "./checkpoint/Encoder_stage2.ckpt" \
    --device_target GPU \
    > log_stage3.txt 2>&1

python -u $BASE_PATH/../eval.py \
  --data_path ${cityscapes_path} \
  --run_distribute false \
  --encode false \
  --model_root_path './checkpoint/ENet_stage3.ckpt' \
  --device_id 1 \
  --device_target GPU \
  > log_eval.txt 2>&1 &

