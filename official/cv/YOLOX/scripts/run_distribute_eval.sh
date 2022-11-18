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

if [ $# != 5 ]
then
    echo "Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [BACKBONE] [BATCH_SIZE] [RANK_TABLE_FILE]"
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
RANK_TABLE_FILE=$(get_real_path $5)
if [ "$BACKBONE" = 'yolox_darknet53' ]
then
  CONFIG_PATH='yolox_darknet53.yaml'
else
  CONFIG_PATH='yolox_x.yaml'
fi
echo $DATA_DIR
echo $VAL_CKPT
echo $PER_BATCH_SIZE
echo $RANK_TABLE_FILE
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

if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file."
exit 1
fi

export DEVICE_NUM=8
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0
export RANK_TABLE_FILE=$RANK_TABLE_FILE

if [ -d "eval_parallel" ];
then
    rm -rf ./eval_parallel
fi
mkdir ./eval_parallel

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`
start_device_id=0

for ((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=$(($start_device_id + $i))
    export RANK_ID=$i
    dir_path=./eval_parallel$i
    if [ -d $dir_path ]
    then
        rm -rf $dir_path
    fi
    mkdir $dir_path
    cp ../*.py $dir_path
    cp ../*.yaml $dir_path
    cp -r ../src $dir_path
    cp -r ../model_utils $dir_path
    cp -r ../third_party $dir_path
    cd $dir_path || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    taskset -c $cmdopt python eval.py \
        --config_path=$CONFIG_PATH\
        --data_dir=$DATA_DIR \
        --backbone=$BACKBONE \
        --val_ckpt=$VAL_CKPT \
        --is_distributed=1 \
        --device_num=$DEVICE_NUM \
        --per_batch_size=$PER_BATCH_SIZE > log.txt 2>&1 &
    cd ..
done
