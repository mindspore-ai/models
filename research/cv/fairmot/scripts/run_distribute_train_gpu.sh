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
        bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [pretrained_model]
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

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

CONFIG=$(get_real_path $3)
echo "CONFIG: "$CONFIG

LOAD_PRE_MODEL=$(get_real_path $4)
echo "PRETRAINED_MODEL: "$LOAD_PRE_MODEL

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -f $LOAD_PRE_MODEL ]
then
    echo "error: pretrained_model=$LOAD_PRE_MODEL is not a file."
exit 1
fi

if [ -d "$BASE_PATH/../train" ];
then
    rm -rf $BASE_PATH/../train
fi
mkdir $BASE_PATH/../train
cd $BASE_PATH/../train || exit

export CUDA_VISIBLE_DEVICES="$2"

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

echo "start training on multiple GPU"
env > env.log
echo
mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python -u ${BASE_PATH}/../train.py --run_distribute True --device GPU --config_path $CONFIG \
      --load_pre_model ${LOAD_PRE_MODEL} &> train.log &
