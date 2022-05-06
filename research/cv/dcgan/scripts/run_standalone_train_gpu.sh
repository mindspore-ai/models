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

if [ $# != 3 ]
then
    echo "Usage: bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_URL] [TRAIN_URL]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ID=$1
echo $ID
PATH1=$2
echo $PATH1
PATH2=$(get_real_path $3)
echo $PATH2

if [ ! -d $PATH1 ]
then
    echo "error: DATA_URL=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: TRAIN_URL=$PATH2 is not a directory"
exit 1
fi

ulimit -c unlimited
export DEVICE_NUM=1
export DEVICE_ID=$ID
export RANK_ID=$ID
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
mkdir ./train/scripts
cp ../*.py ./train
cp *.sh ./train/scripts
cp -r ../src ./train
cp -r ../gpu_infer ./train
cd ./train || exit
echo "start training for device $ID"
env > env.log
nohup python -u train.py --device_target=GPU --device_id=$ID --data_url=$PATH1 --train_url=$PATH2 > output.train_log 2>&1 &
cd ..
