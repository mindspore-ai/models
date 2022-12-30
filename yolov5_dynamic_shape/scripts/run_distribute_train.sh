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

if [ $# != 1 ]
then
    echo "Usage: bash run_distribute_train.sh [DATASET_PATH]"
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

export RANK_SIZE=8


rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp ../*.yaml ./train_parallel
cp -r ../src ./train_parallel
cp -r ../model_utils ./train_parallel
cd ./train_parallel || exit
echo "start training for $RANK_SIZE p."
env > env.log
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
python train.py \
    --data_dir=$DATASET_PATH \
    --is_distributed=1 \
    --lr=0.02 \
    --per_batch_size=16 > log.txt 2>&1 &
cd ..
