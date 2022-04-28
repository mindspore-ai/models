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

if [ $# != 1 ]
then 
    echo "Usage: bash run_distribute_train_gpu.sh [DATASET_PATH]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)

if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
    exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp *.sh ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit

mpirun --allow-run-as-root -n $RANK_SIZE \
        python train.py --run_distribute=True --device_num=$DEVICE_NUM \
        --device_target="GPU" --dataset_path=$PATH1 > autoslim_gpu.log 2>&1 &

cd ..
