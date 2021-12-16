#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [ $# != 5 ]
then 
    echo "Usage: bash run_distribute_train_gpu.sh [CONFIG_FILE] [DATASET] [DEVICE_NUM] [EPOCH_SIZE] [LR]"
    exit 1
fi


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

CONFIG_PATH=$(get_real_path "$1")
DATASET_PATH=$2
DEVICE_NUM_=$3
EPOCH_SIZE=$4
LR=$5

if [[ ! -f $CONFIG_PATH ]]
then
    echo "error: CONFIG_FILE=$CONFIG_PATH is not a file"
exit 1
fi

cd .. || exit 1
python train.py --only_create_dataset=True --distribute=True  \
                            --config_path="$CONFIG_PATH" \
                            --device_num="$DEVICE_NUM_" \
                            --run_platform="GPU" \
                            --dataset="$DATASET_PATH" \
                            --lr="$LR" \
                            --epoch_size="$EPOCH_SIZE"

cd scripts || exit 1


#ulimit -u unlimited
export DEVICE_NUM=$DEVICE_NUM_
export RANK_SIZE=$DEVICE_NUM

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp -- *.sh ./train_parallel
cp -r ../src ./train_parallel
cp -r ../config/*.yaml ./train_parallel
cd ./train_parallel || exit

if [ $# == 5 ]
then
  mpirun \
  --allow-run-as-root \
  -n "$RANK_SIZE" \
  --output-filename log_output \
  --merge-stderr-to-stdout python train.py \
                            --distribute=True  \
                            --config_path="$CONFIG_PATH" \
                            --device_num="$DEVICE_NUM" \
                            --run_platform="GPU" \
                            --dataset="$DATASET_PATH" \
                            --lr="$LR" \
                            --epoch_size="$EPOCH_SIZE" &> log &
fi
