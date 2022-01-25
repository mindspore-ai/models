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

# applicable to Ascend

if [ $# != 3 ]
then 
    echo "Usage: bash run_distribute_train_ascend.sh [PREPROCESSED_TRAINING_DATASET_PATH] [RANK_SIZE] [RANK_TABLE_FILE]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}


DATA_PATH=$(get_real_path $1)
RANK_SIZE=$2
RANK_TABLE_FILE=$(get_real_path $3)

export GLOG_v=3

if [ ! -d $DATA_PATH ]
then 
  echo "error: PREPROCESSED_TRAINING_DATASET_PATH=$DATA_PATH is not a directory"
  exit 1
fi

if [ ! -f $RANK_TABLE_FILE ]
then 
  echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
  exit 1
fi

export RANK_TABLE_FILE=$RANK_TABLE_FILE
export RANK_SIZE=$RANK_SIZE
export DATA_PATH=${DATA_PATH}

cd ..

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf train_results/device$i
    mkdir -p train_results/device$i
    cp -r src/ ./train_results/device$i
    cp train_lpcnet_parallel.py ./train_results/device$i
    cd ./train_results/device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "info: start training for device $i"
    env > env$i.log
    python -u train_lpcnet_parallel.py "$DATA_PATH/features.f32" "$DATA_PATH/data.s16" "./ckpt$i" > train.log$i 2>&1 &
    cd ../../
done
