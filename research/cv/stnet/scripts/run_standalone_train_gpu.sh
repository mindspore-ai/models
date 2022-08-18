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

if [ $# != 2 ] && [ $# != 3 ]
then
  echo "Usage: 
        bash run_standalone_train_gpu.sh [CONFIG] [DATASET] [PRETRAINED_CKPT_PATH](optional)
       "
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG=$(get_real_path $1)
echo "CONFIG: "$CONFIG

DATASET=$(get_real_path $2)
echo "DATASET: "$DATASET

if [ $# == 3 ]
then
    PRETRAINED_CKPT_PATH=$(get_real_path $3)
    echo "PRETRAINED_CKPT_PATH: $PRETRAINED_CKPT_PATH"
else
    PRETRAINED_CKPT_PATH=''
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -d $DATASET ]
then
    echo "error: dataset=$DATASET is not a directory."
exit 1
fi

if [ $# == 3 ] && [ ! -f $PRETRAINED_CKPT_PATH ]
then
    echo "error: pretrained_ckpt_path=$PRETRAINED_CKPT_PATH is not a file."
exit 1
fi

if [ -d "$BASE_PATH/../train" ];
then
    rm -rf $BASE_PATH/../train
fi
mkdir $BASE_PATH/../train
cd $BASE_PATH/../train || exit

echo "Training on GPU..."
echo
env > env.log
pwd
echo
python -u ${BASE_PATH}/../train.py --target=GPU --config_path=$CONFIG --dataset_path=$DATASET --pre_res50=$PRETRAINED_CKPT_PATH &> train.log &
