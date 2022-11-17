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

if [ $# != 2 ]; then
  echo "Usage: 
        bash run_standalone_train_ascend.sh [config_file] [dataset_path]
       " 
  exit 1
fi

get_real_path() {
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

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

if [ -d "../train" ];
then
    rm -rf ../train
fi
mkdir ../train
cd ../train || exit

echo "Training on Ascend..."
echo
env > env.log
pwd
echo
nohup python ${BASE_PATH}/../train.py --device_target=Ascend --config_path=$CONFIG --train_data=$DATASET &> train.log &