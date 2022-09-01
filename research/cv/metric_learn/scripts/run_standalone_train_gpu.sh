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

if [ $# != 3 ] && [ $# != 4 ]; then
  echo "Usage: bash run_standalone_train_gpu.sh [DATASET_PATH] [DEVICE_ID] [LOSS_NAME] [CHECKPOINT_PATH](optional)"
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

if [ $# == 4 ]
then
    PATH2=$(get_real_path $4)
fi

if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ $# == 4 ] && [ ! -f $PATH2 ]
then
    echo "error: CKPT_PATH=$PATH2 is not a file"
exit 1
fi

ulimit -u unlimited
FOLDER_NAME='train_'$3

rm -rf $FOLDER_NAME
mkdir $FOLDER_NAME
cp ../*.py $FOLDER_NAME
cp *.sh $FOLDER_NAME
cp -r ../src $FOLDER_NAME
cd $FOLDER_NAME || exit
env > env.log
echo "start training for GPU $2"

if [ $# == 4 ]
then
    python train.py --device_target="GPU" --dataset_path=$PATH1 --device_id=$2 \
                    --loss_name=$3 --ckpt_path=$PATH2 --run_eval=False > log 2>&1 &
fi

if [ $# == 3 ]
then
    python train.py --device_target="GPU" --dataset_path=$PATH1 --device_id=$2 \
                    --loss_name=$3 --run_eval=False > log 2>&1 &
fi

cd ..
