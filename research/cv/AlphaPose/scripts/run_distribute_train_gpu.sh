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

if [ $# != 5 ]; then
  echo "Usage: 
        bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [dataset_dir] [pretrained_backbone]
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

DATASET=$(get_real_path $4)
echo "DATASET: "$DATASET

BACKBONE=$(get_real_path $5)
echo "BACKBONE: "$BACKBONE

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -d $DATASET ]
then
    echo "error: dataset_root=$DATASET is not a directory."
exit 1
fi

if [ ! -f $BACKBONE ]
then
    echo "error: pretrained_backbone=$BACKBONE is not a file."
exit 1
fi

if [ -d "$BASE_PATH/../train_parallel" ];
then
    rm -rf $BASE_PATH/../train_parallel
fi
mkdir $BASE_PATH/../train_parallel
cd $BASE_PATH/../train_parallel || exit

export CUDA_VISIBLE_DEVICES="$2"

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

echo "start training on multiple GPUs"
env > env.log
echo
mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
  python -u ${BASE_PATH}/../train.py --DEVICE_TARGET GPU --RUN_DISTRIBUTE True \
    --config_path $CONFIG --DATASET_ROOT $DATASET --MODEL_PRETRAINED $BACKBONE &> train.log &
