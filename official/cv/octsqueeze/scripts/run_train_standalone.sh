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

if [ $# != 4 ]
then 
    echo "Usage: bash run_train_standalone.sh [TRAINING_DATASET_PATH] [DEVICE] [CHECKPOINT_SAVE_PATH] [batch_size]"
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
    echo "error: TRAINING_DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

data_path=$1
device_target=$2
checkpoint=$3
batch_size=$4

python ../train.py --train=$data_path --device_target=$device_target --checkpoint=$checkpoint --batch_size=$batch_size --is_distributed=0
