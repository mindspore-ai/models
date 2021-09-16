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

if [ $# != 4 ]; then
  echo "Usage: bash ./run_distribute_train.sh [RANK_TABLE_FILE] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [TRAIN_DATA_DIR] [EVAL_DATA_DIR]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

RANK_TABLE_FILE=$(get_real_path $1)
echo $RANK_TABLE_FILE

if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file."
exit 1
fi

VISIABLE_DEVICES=$2
IFS="," read -r -a CANDIDATE_DEVICE <<< "$VISIABLE_DEVICES"
export RANK_SIZE=${#CANDIDATE_DEVICE[@]}

TRAIN_DATA_DIR=$(get_real_path $3)
echo $TRAIN_DATA_DIR

if [ ! -d $TRAIN_DATA_DIR ]
then
    echo "error: TRAIN_DATA_DIR=$TRAIN_DATA_DIR is not a directory."
exit 1
fi

EVAL_DATA_DIR=$(get_real_path $4)
echo $EVAL_DATA_DIR

if [ ! -d $EVAL_DATA_DIR ]
then
    echo "error: EVAL_DATA_DIR=$EVAL_DATA_DIR is not a directory."
exit 1
fi

export RANK_TABLE_FILE=$RANK_TABLE_FILE

cores=`cat /proc/cpuinfo|grep "processor" |wc -l`
echo "the number of logical core" $cores
avg_core_per_rank=`expr $cores \/ $RANK_SIZE`
core_gap=`expr $avg_core_per_rank \- 1`
echo "avg_core_per_rank" $avg_core_per_rank
echo "core_gap" $core_gap
for((i=0;i<RANK_SIZE;i++))
do
    start=`expr $i \* $avg_core_per_rank`
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    export RANK_ID=$i
    export DEPLOY_MODE=0
    export GE_USE_STATIC_MEMORY=1
    end=`expr $start \+ $core_gap`
    cmdopt=$start"-"$end

    rm -rf train_parallel$i
    mkdir ./train_parallel$i
    cp  *.py ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $i, device $DEVICE_ID rank_id $RANK_ID"

    env > env.log
    taskset -c $cmdopt python -u ../train.py \
    --train_dataset_path=$TRAIN_DATA_DIR --eval_dataset_path=$EVAL_DATA_DIR --device_id=${CANDIDATE_DEVICE[i]}  > log.txt 2>&1 &
    cd ../
done
