#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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
    echo "Usage: bash run_standalone_train_gpu.sh [DATASET_PATH] [DEVICE_ID]"
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

DEVICE_ID=0
if [ $# == 2 ]; then
    DEVICE_ID=$2
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp -r ../src ./train
cp -r ../model_utils ./train
cp -r ../third_party ./train
cd ./train || exit
echo "======start training======"
env > env.log

python train.py --data_dir=$DATASET_PATH --device_target=GPU > log.txt 2>&1 &
cd ..
