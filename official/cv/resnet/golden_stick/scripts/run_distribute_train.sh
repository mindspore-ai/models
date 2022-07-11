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

CURPATH="$(dirname "$0")"

if [ $# != 4 ] && [ $# != 6 ]
then
  echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [PYTHON_PATH] [CONFIG_PATH] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)"
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
PYTHON_PATH=$(get_real_path $2)
CONFIG_FILE=$(get_real_path $3)
PATH2=$(get_real_path $4)

if [ ! -d $PYTHON_PATH ]
then
    echo "error: PYTHON_PATH=$PYTHON_PATH is not a directory"
    exit 1
fi


if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -d $PATH2 ]
then 
    echo "error: DATASET_PATH=$PATH2 is not a directory"
exit 1
fi 

if [ $# == 6 ]; then
  CKPT_TYPE=$5
  CKPT_FILE=$(get_real_path $6)

  if [ "x$CKPT_TYPE" != "xFP32" ] && [ "x$CKPT_TYPE" != "xPRETRAINED" ]; then
      echo "error: CKPT_TYPE=$CKPT_TYPE is not valid, should be FP32 or PRETRAINED"
      exit 1
  fi
  if [ ! -f $CKPT_FILE ]; then
      echo "error: CKPT_PATH=$CKPT_FILE is not a file"
      exit 1
  fi
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ${PYTHON_PATH}/*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ${CURPATH}/../../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    
    if [ "x$CKPT_TYPE" == "xFP32" ]; then
        taskset -c $cmdopt python3 train.py --run_distribute=True --device_num=$RANK_SIZE --data_path=$PATH2 \
        --config_path=$CONFIG_FILE --fp32_ckpt=$CKPT_FILE --output_path './output' &> log &
    elif [ "x$CKPT_TYPE" == "xPRETRAINED" ]; then
        taskset -c $cmdopt python3 train.py --run_distribute=True --device_num=$RANK_SIZE --data_path=$PATH2 \
        --config_path=$CONFIG_FILE --pre_trained=$CKPT_FILE --output_path './output' &> log &
    else
        taskset -c $cmdopt python3 train.py --run_distribute=True --device_num=$RANK_SIZE --data_path=$PATH2 \
        --config_path=$CONFIG_FILE --output_path './output' &> log &
    fi
    cd ..
done
