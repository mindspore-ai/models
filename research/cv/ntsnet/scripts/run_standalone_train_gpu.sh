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

if [ $# -lt 2 ]
then
    echo "Usage: bash run_standalone_train_gpu.sh [DATA_URL] [TRAIN_URL] [DEVICE_ID(optional)]"
exit 1
fi

export DEVICE_ID=0

if [ $# = 3 ] ; then
  export DEVICE_ID=$3
fi;


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$(get_real_path $1)
echo $PATH1
PATH2=$(get_real_path $2)
echo $PATH2

if [ ! -d $PATH1 ]
then
    echo "error: DATA_URL=$PATH1 is not a directory"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train$3
fi
mkdir ./train$3
cp ../*.py ./train$3
cp *.sh ./train$3
cp -r ../src ./train$3
cd ./train$3 || exit
echo "start training for device $DEVICE_ID"
env > env.log
python train_gpu.py --device_id=$DEVICE_ID --data_url=$PATH1 --train_url=$PATH2 --device_target="GPU"&> log &
cd ..
