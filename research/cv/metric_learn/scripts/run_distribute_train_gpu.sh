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

if [ $# != 2 ] && [ $# != 3 ]; then
  echo "Usage: bash run_distribute_train_gpu.sh [DATASET_PATH] [LOSS_NAME] [CHECKPOINT_PATH](optional)"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)

if [ $# == 3 ]
then
    PATH2=$(get_real_path $3)
fi

if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ $# == 3 ] && [ ! -f $PATH2 ]
then
    echo "error: CKPT_PATH=$PATH2 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8

FOLDER_NAME='train_parallel_'$2

rm -rf $FOLDER_NAME
mkdir $FOLDER_NAME
cp ../*.py $FOLDER_NAME
cp *.sh $FOLDER_NAME
cp -r ../src $FOLDER_NAME
cd $FOLDER_NAME || exit
env > env.log
echo "start parallel training on $DEVICE_NUM GPUs"

if [ $# == 3 ]
then
    mpirun --allow-run-as-root -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout \
            python train.py --run_distribute=True --device_target="GPU" --dataset_path=$PATH1 \
            --loss_name=$2 --ckpt_path=$PATH2 --run_eval=False > log 2>&1 &
fi

if [ $# == 2 ]
then
    mpirun --allow-run-as-root -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout \
            python train.py --run_distribute=True --device_target="GPU" --dataset_path=$PATH1 \
            --loss_name=$2 --run_eval=False > log 2>&1 &
fi

cd ..
