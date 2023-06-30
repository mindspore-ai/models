#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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
if [ $# != 3 ] && [ $# != 4 ] && [ $# != 5 ] && [ $# != 6 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE]"
  echo "bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional)"
  echo "bash run_standalone_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional)"
  echo "bash run_standalone_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional) [RUN_EVAL](optional)"
  echo "For example: bash run_distribute_train.sh 8 /path/hccl_8p.json /segformer/config/segformer.b0.512x1024.city.yaml /segformer/data/cityscapes/"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DEVICE_NUM=$1
RANK_TABLE_FILE=$(get_real_path $2)
CONFIG_FILE=$(get_real_path $3)
RANK_SIZE=${DEVICE_NUM}

if [ $# == 4 ]
then
  DATASET_PATH=$(get_real_path $4)
fi

if [ $# == 5 ]
then
  DATASET_PATH=$(get_real_path $4)
  PRETRAINED_CKPT_PATH=$(get_real_path $5)
fi

if [ $# == 6 ]
then
  DATASET_PATH=$(get_real_path $4)
  PRETRAINED_CKPT_PATH=$(get_real_path $5)
  RUN_EVAL=$6
fi

echo "DEVICE_NUM=${DEVICE_NUM}"
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "PRETRAINED_CKPT_PATH=${PRETRAINED_CKPT_PATH}"
echo "RUN_EVAL=${RUN_EVAL}"

export PYTHONUNBUFFERED=1
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
export RANK_SIZE=${RANK_SIZE}
export DEVICE_NUM=${RANK_SIZE}

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

for((i=0;i<${RANK_SIZE};i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    work_dir=run_distribute_train_device$i
    rm -rf $work_dir
    mkdir $work_dir

    if [ $# == 3 ]
    then
      cp ../*.py ../src ../config ../data ../pretrained ./$work_dir -r
      cd ./$work_dir
      echo "start standalone training for device $DEVICE_ID"
      env > env.log
      taskset -c $cmdopt python train.py --config_path=$CONFIG_FILE --run_distribute=True --batch_size=16 > train.log 2>&1 &
    fi

    if [ $# == 4 ]
    then
      cp ../*.py ../src ../config ../pretrained ./$work_dir -r
      cd ./$work_dir
      echo "start standalone training for device $DEVICE_ID"
      env > env.log
      taskset -c $cmdopt python train.py --config_path=$CONFIG_FILE --data_path=$DATASET_PATH --run_distribute=True --batch_size=16 > train.log 2>&1 &
    fi

    if [ $# == 5 ]
    then
      cp ../*.py ../src ../config ./$work_dir -r
      cd ./$work_dir
      echo "start standalone training for device $DEVICE_ID"
      env > env.log
      taskset -c $cmdopt python train.py --config_path=$CONFIG_FILE --data_path=$DATASET_PATH --pretrained_ckpt_path=$PRETRAINED_CKPT_PATH --run_distribute=True --batch_size=16 > train.log 2>&1 &
    fi

    if [ $# == 6 ]
    then
      cp ../*.py ../src ../config ./$work_dir -r
      cd ./$work_dir
      echo "start standalone training for device $DEVICE_ID"
      env > env.log
      taskset -c $cmdopt python train.py --config_path=$CONFIG_FILE --data_path=$DATASET_PATH --pretrained_ckpt_path=$PRETRAINED_CKPT_PATH --run_eval=$RUN_EVAL --run_distribute=True --batch_size=16 > train.log 2>&1 &
    fi

    cd ../
done
