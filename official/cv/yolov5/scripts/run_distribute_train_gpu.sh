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

if [ $# != 2 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
echo $DATASET_PATH

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

export RANK_SIZE=$2

if [ -d "distribute_train" ]; then
  rm -rf ./distribute_train
fi

mkdir ./distribute_train
cp ../*.py ./distribute_train
cp ../*.yaml ./distribute_train
cp -r ../src ./distribute_train
cp -r ../model_utils ./distribute_train
cd ./distribute_train || exit

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python train.py \
      --device_target=GPU \
      --per_batch_size=32 \
      --data_dir=$DATASET_PATH \
      --is_distributed=1 \
      --yolov5_version='yolov5s' \
      --lr=0.025 \
      --T_max=300 \
      --max_epoch=300 \
      --warmup_epochs=20 \
      --training_shape=640 \
      --lr_scheduler=cosine_annealing > log.txt 2>&1 &
cd ..
