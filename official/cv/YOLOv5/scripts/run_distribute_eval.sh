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

if [ $# != 3 ]
then
    echo "Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [RANK_TABLE_FILE]"
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
CHECKPOINT_PATH=$(get_real_path $2)
RANK_TABLE_FILE=$(get_real_path $3)
echo $DATASET_PATH
echo $CHECKPOINT_PATH

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $CHECKPOINT_PATH ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

if [ ! -f $RANK_TABLE_FILE ]
then
  echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a valid file."
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
  if [ -d $dir_path ]; then
    rm -rf $dir_path
  fi
  mkdir $dir_path
  cp ../*.py $dir_path
  cp ../*.yaml $dir_path
  cp -r ../src $dir_path
  cp -r ../model_utils $dir_path
  cp -r ../third_party $dir_path
  cd $dir_path || exit
  env > env.log
  echo "start inferring for rank $RANK_ID, device $DEVICE_ID"
  taskset -c $cmdopt python eval.py \
      --is_distributed=1 \
      --data_dir=$DATASET_PATH \
      --pretrained=$CHECKPOINT_PATH > log.txt 2>&1 &
  cd ..
done
