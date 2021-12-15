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

if [ $# != 3 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM] [CPUS_PER_RANK]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path "$1")
DEVICE_NUM=$2
CPUS_PER_RANK=$3
echo "$DATASET_PATH"

if [ ! -d "$DATASET_PATH" ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp ../*.yaml ./train_parallel
cp -r ../src ./train_parallel
cp -r ../model_utils ./train_parallel
cd ./train_parallel || exit
env > env.log
mpirun --allow-run-as-root -n "$DEVICE_NUM" --cpus-per-rank "$CPUS_PER_RANK" --output-filename log_output --merge-stderr-to-stdout \
python train.py \
    --data_dir="$DATASET_PATH" \
    --device_target=GPU \
    --is_distributed=1 \
    --lr=0.002 \
    --t_max=300 \
    --max_epoch=300 \
    --warmup_epochs=4 \
    --training_shape=640 \
    --lr_scheduler=cosine_annealing  > log.txt 2>&1 &
cd ..
