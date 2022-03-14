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
  echo "Usage: bash scripts/run_distribute_train_gpu.sh  RANK_SIZE CUDA_VISIBLE_DEVICES /path/to/cityscapes"
  echo "Example: bash scripts/run_distribute_train_gpu.sh 4 0,1,2,3 /home/name/cityscapes"
  exit 1
fi

if [ ! -d $3 ]
then
    echo "error: DATASET_PATH=$3 is not a directory"
exit 1
fi

# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)



echo "RANK_SIZE: $1"
echo "CUDA_VISIBLE_DEVICES: $2"
echo "cityscapes_path: $3"
export RANK_SIZE=$1
export CUDA_VISIBLE_DEVICES=$2
cityscapes_path=$3

mkdir ./log_distribute_device
cd ./log_distribute_device
mkdir ./checkpoint

# 1.train
  echo "start training stage1"

mpirun -n $RANK_SIZE --output-filename log_output1 --merge-stderr-to-stdout --allow-run-as-root \
  python -u $BASE_PATH/../train.py \
    --lr 1e-3 \
    --repeat 2 \
    --run_distribute true \
    --save_path './checkpoint' \
    --mindrecord_train_data "$BASE_PATH/../data/train.mindrecord" \
    --stage 1 \
    --ckpt_path "" \
    --device_target GPU \
    > log_stage1.txt 2>&1
wait
# 2.train
  echo "start training stage2"

mpirun -n $RANK_SIZE --output-filename log_output2 --merge-stderr-to-stdout --allow-run-as-root \
  python -u $BASE_PATH/../train.py \
    --lr 1e-3 \
    --repeat 2 \
    --run_distribute true \
    --save_path './checkpoint' \
    --mindrecord_train_data "$BASE_PATH/../data/train.mindrecord" \
    --stage 2 \
    --ckpt_path "./checkpoint/Encoder_stage1.ckpt" \
    --device_target GPU \
    > log_stage2.txt 2>&1
wait

# 3.train
  echo "start training stage3"

mpirun -n $RANK_SIZE --output-filename log_output3 --merge-stderr-to-stdout --allow-run-as-root \
  python -u $BASE_PATH/../train.py \
    --lr 1e-3 \
    --repeat 2 \
    --run_distribute true \
    --save_path './checkpoint' \
    --mindrecord_train_data "$BASE_PATH/../data/train.mindrecord" \
    --stage 3 \
    --ckpt_path "./checkpoint/Encoder_stage2.ckpt" \
    --device_target GPU \
    > log_stage3.txt 2>&1
wait

# 4.eval
  echo "start evaling"

python -u $BASE_PATH/../eval.py \
  --data_path ${cityscapes_path} \
  --run_distribute false \
  --encode false \
  --model_root_path './checkpoint/ENet_stage3.ckpt' \
  --device_id 1 \
  --device_target GPU \
  > log_eval.txt 2>&1 &

