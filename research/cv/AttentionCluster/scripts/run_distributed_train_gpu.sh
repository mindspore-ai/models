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

if [ $# != 4 ]; then
    echo "Usage: 
          bash run_distributed_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [CONFIG] [DATASET_DIR]
         "
exit 1
fi

get_real_path(){
  if [ -z $1 ]; then
    echo "error: DATASET_DIR is empty"
    exit 1
  elif [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export CUDA_VISIBLE_DEVICES="$2"

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

CONFIG=$(get_real_path $3)
DATASET_DIR=$(get_real_path $4)

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -d $DATASET_DIR ]
then
    echo "error: DATASET_PATH=$DATASET_DIR is not a directory"
exit 1
fi

echo "CONFIG: $CONFIG"
echo "DATASET_DIR: $DATASET_DIR"
echo

if [ -d "$BASE_PATH/../train" ];
then
    rm -rf $BASE_PATH/../train
fi
mkdir $BASE_PATH/../train
cd $BASE_PATH/../train || exit

mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
  python ${BASE_PATH}/../train.py --distributed True --device GPU --config_path $CONFIG --data_dir $DATASET_DIR --result_dir . &> train.log &
