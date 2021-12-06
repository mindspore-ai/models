#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: sh run_train_distribute_ascend.sh [RANK_TABLE_FILE] [market1501|dukemtmcreid|cuhk03|msmt17] [PRETRAINED_CKPT_PATH](optional)"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi


if [ $2 != "market1501" ] && [ $2 != "dukemtmcreid" ] && [ $2 != "cuhk03" ] && [ $2 != "msmt17" ]
then
    echo "error: the selected dataset is not market1501, dukemtmcreid, cuhk03 or msmt17"
exit 1
fi
dataset_name=$2

if [ $# == 3 ]
then
  if [ ! -f $3 ]
  then
    echo "error: PRETRAINED_CKPT_PATH=$3 is not a file"
  exit 1
  fi
  PATH2=$(realpath $3)
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
PATH1=$(realpath $1)

export RANK_TABLE_FILE=$PATH1
echo "RANK_TABLE_FILE=${PATH1}"

EXECUTE_PATH=$(pwd)
config_path="${EXECUTE_PATH}/../osnet_config.yaml"
echo "config path is : ${config_path}"

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
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cp -r ../model_utils ./train_parallel$i
    cp -r ../*.yaml ./train_parallel$i
    cp ../train.py ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID, $dataset_name"
    cd ./train_parallel$i ||exit
    env > env.log
    if [ $# == 2 ]
    then
      taskset -c $cmdopt python train.py --config_path=$config_path --source=$dataset_name \
      --run_distribute=True --output_path='./output'> train.log 2>&1 &
    fi

    if [ $# == 3 ]
    then
      taskset -c $cmdopt python train.py --config_path=$config_path --source=$dataset_name \
      --checkpoint_file_path=$PATH2 --run_distribute=True --output_path='./output'> train.log 2>&1 &
    fi
    cd ..
done
