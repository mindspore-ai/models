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
# ===========================================================================

if [ $# != 3 ]
then
    echo "Usage: bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE] [DEVICE_NUM]"
exit 1
fi
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
RANK_TABLE_FILE=$(get_real_path $2)
DEVICE_NUM=$3
if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if [ ! -f $PRETRAINED_BACKBONE ]
then
    echo "error: PRETRAINED_PATH=$PRETRAINED_BACKBONE is not a file"
exit 1
fi

if [ $DEVICE_NUM = 4 ]||[ $DEVICE_NUM != 8 ]
then
    echo "error: DEVICE_NUM=$DEVICE_NUM must be 4 or 8"
exit 1
fi

if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
exit 1
fi
export DEVICE_NUM=$DEVICE_NUM
export RANK_SIZE=$DEVICE_NUM
export RANK_TABLE_FILE=$RANK_TABLE_FILE


cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`


for((i=0; i<${DEVICE_NUM}; i++))
do

    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    echo "start" $start
    echo "end" $end
    echo "cmdopt" $cmdopt
    export DEVICE_ID=$i
    export RANK_ID=$i

    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i

    cp ./*.py ./train_parallel$i
    cp -r ./model_utils ./train_parallel$i
    cp -r ./ascend310_infer ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cp -r ./scripts ./train_parallel$i

    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log

    taskset -c $cmdopt python train.py\
    --dataset_path=$DATASET_PATH\
    --distribute=true >log 2>&1 &
    cd ..
done
