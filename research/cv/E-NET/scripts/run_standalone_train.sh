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


if [ $# != 2 ]
then
  echo "Usage: bash scripts/run.sh /path/to/cityscapes DEVICE_ID"
  echo "Example: bash scripts/run.sh /home/name/cityscapes 0"
  exit 1
fi

if [ ! -d $1 ]
then 
    echo "error: DATASET_PATH=$1 is not a directory"
exit 1
fi

echo "CityScapes dataset path: $1"
echo "DEVICE_ID: $2"

ps -aux | grep "python -u ../train.py" | awk '{print $2}' | xargs kill -9

mkdir ./log_single_device
cd ./log_single_device

cityscapes_path=$1
export RANK_SIZE=1
export DEVICE_ID=$2

python -u ../train.py \
    --lr 5e-4 \
    --repeat 1 \
    --run_distribute false \
    --save_path './' \
    --mindrecord_train_data "../data/train.mindrecord" \
    --stage 1 \
    --ckpt_path "" \
    > log_stage1.txt 2>&1

python -u ../train.py \
    --lr 5e-4 \
    --repeat 1 \
    --run_distribute false \
    --save_path './' \
    --mindrecord_train_data "../data/train.mindrecord" \
    --stage 2 \
    --ckpt_path "./Encoder-65_496.ckpt" \
    > log_stage2.txt 2>&1

python -u ../train.py \
    --lr 5e-4 \
    --repeat 1 \
    --run_distribute false \
    --save_path './' \
    --mindrecord_train_data "../data/train.mindrecord" \
    --stage 3 \
    --ckpt_path "./Encoder_1-85_496.ckpt" \
    > log_stage3.txt 2>&1

python -u ../eval.py \
  --data_path ${cityscapes_path} \
  --run_distribute false \
  --encode false \
  --model_root_path './' \
  --device_id ${DEVICE_ID} \
  > log_eval.txt 2>&1 &