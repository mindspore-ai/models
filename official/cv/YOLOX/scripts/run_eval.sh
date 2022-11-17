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

if [ $# != 4 ]
then
    echo "Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [BACKBONE] [BATCH_SIZE] "
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_DIR=$(get_real_path $1)
VAL_CKPT=$(get_real_path $2)
BACKBONE=$3
PER_BATCH_SIZE=$4
if [ "$BACKBONE" = 'yolox_darknet53' ]
then
  CONFIG_PATH='yolox_darknet53.yaml'
else
  CONFIG_PATH='yolox_x.yaml'
fi
echo $DATA_DIR
echo $VAL_CKPT
echo $PER_BATCH_SIZE
if [ ! -d $DATA_DIR ]
then
    echo "error: DATASET_PATH=$DATA_DIR is not a directory"
exit 1
fi

if [ ! -f $VAL_CKPT ]
then
    echo "error: CHECKPOINT_PATH=$VAL_CKPT is not a file"
exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cp -r ../model_utils ./eval
cd ./eval || exit
env > env.log
echo "start inferring for device $DEVICE_ID"
python eval.py \
    --config_path=$CONFIG_PATH \
    --data_dir=$DATA_DIR \
    --val_ckpt=$VAL_CKPT \
    --backbone=$BACKBONE \
    --per_batch_size=$PER_BATCH_SIZE  > log.txt 2>&1 &
cd ..