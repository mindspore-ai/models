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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 3 ]
then
    echo "Usage: sh run_distributution_gpu.sh [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [TRAIN_MODEL]"
exit 1
fi


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
TRAIN_PATH=$(get_real_path $1)
EVAL_PATH=$(get_real_path $2)
TRAIN_MODEL=$3
export RANK_SIZE=8

echo $TRAIN_PATH
echo $EVAL_PATH
echo $TRAIN_MODEL

if [ ! -d $PATH1 ]
then
    echo "error: TRAIN_DATA_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: EVAL_DATA_PATH=$PATH2 is not a directory"
exit 1
fi

rm -rf device
mkdir device
cp -r ../src/ ./device
cp ../train.py ./device
echo "start training"
cd ./device

mpirun --allow-run-as-root -n 8 python train.py --device_target=GPU --is_distributed=1 --train_path=$TRAIN_PATH --eval_path=$EVAL_PATH  --train_model=$TRAIN_MODEL > log 2>&1 &
