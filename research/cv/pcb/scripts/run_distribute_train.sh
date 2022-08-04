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

if [ $# != 5 ] && [ $# != 6 ]
then 
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)"
    exit 1
fi


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

RANK_TABLE_FILE=$(get_real_path $1)
MODEL_NAME=$2
DATASET_NAME=$3
DATASET_PATH=$(get_real_path $4)
CONFIG_PATH=$(get_real_path $5)

if [ $# == 6 ]; then
  PRETRAINED_CKPT_PATH=$(get_real_path $6)
else
  PRETRAINED_CKPT_PATH=""
fi

if [ ! -f $RANK_TABLE_FILE ]
then 
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
exit 1
fi

if [ $MODEL_NAME != "PCB" ] && [ $MODEL_NAME != "RPP" ]
then 
    echo "error: MODEL_NAME=$MODEL_NAME is invalid, please choose from ['PCB','RPP']"
exit 1
fi

if [ $DATASET_NAME != "market" ] && [ $DATASET_NAME != "duke" ] && [ $DATASET_NAME != "cuhk03" ]
then 
    echo "error: DATASET_NAME=$DATASET_NAME is invalid, please choose from ['market','duke','cuhk03']"
exit 1
fi

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if ( [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "market" ] ) || ( [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "duke" ] )
then
    if [ ! -f $CONFIG_PATH ]
    then 
        echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
    exit 1
    fi
else
    if [ ! -d $CONFIG_PATH ]
    then 
        echo "error: CONFIG_PATH=$CONFIG_PATH is not a directory"
    exit 1
    fi
fi

if [ $# == 6 ] && [ ! -f $PRETRAINED_CKPT_PATH ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PRETRAINED_CKPT_PATH is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_FILE

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

script_path=$(readlink -f "$0")
script_dir_path=$(dirname "${script_path}")

LOG_SAVE_PATH=${script_dir_path}/output/log/${MODEL_NAME}/${DATASET_NAME}/train/
CHECKPOINT_SAVE_PATH=${script_dir_path}/output/checkpoint/${MODEL_NAME}/${DATASET_NAME}/train/

if [ -d ${LOG_SAVE_PATH} ];
then
    rm -rf ${LOG_SAVE_PATH}
fi

if [ -d ${CHECKPOINT_SAVE_PATH} ];
then
    rm -rf ${CHECKPOINT_SAVE_PATH}
fi

distribute_train(){
    for((i=0; i<${DEVICE_NUM}; i++))
    do
        start=`expr $i \* $avg`
        end=`expr $start \+ $gap`
        cmdopt=$start"-"$end
        export DEVICE_ID=${i}
        export RANK_ID=$((rank_start + i))
        echo "start training for rank $RANK_ID, device $DEVICE_ID"
        taskset -c $cmdopt python ${script_dir_path}/../train.py --run_distribute=True --device_num=$RANK_SIZE --dataset_path=$DATASET_PATH --config_path=$1 --checkpoint_file_path=$2 --output_path=${script_dir_path}/output/ &
    done
}

if [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "market" ]; then
    distribute_train $CONFIG_PATH $PRETRAINED_CKPT_PATH
elif [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "duke" ]; then
    distribute_train $CONFIG_PATH $PRETRAINED_CKPT_PATH
elif [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "cuhk03" ]; then
    distribute_train $CONFIG_PATH/train_PCB.yaml $PRETRAINED_CKPT_PATH
    wait
    distribute_train $CONFIG_PATH/finetune_PCB.yaml ${CHECKPOINT_SAVE_PATH}/ckpt_0/PCB-40_14.ckpt
elif [ $MODEL_NAME = "RPP" ] && [ $DATASET_NAME = "market" ]; then
    distribute_train ${CONFIG_PATH}/train_PCB.yaml $PRETRAINED_CKPT_PATH
    wait
    distribute_train ${CONFIG_PATH}/train_RPP.yaml ${CHECKPOINT_SAVE_PATH}/ckpt_0/PCB-20_25.ckpt
    wait
    distribute_train ${CONFIG_PATH}/finetune_RPP.yaml ${CHECKPOINT_SAVE_PATH}/ckpt_0/RPP-45_25.ckpt
elif [ $MODEL_NAME = "RPP" ] && [ $DATASET_NAME = "duke" ]; then
    distribute_train ${CONFIG_PATH}/train_PCB.yaml $PRETRAINED_CKPT_PATH
    wait
    distribute_train ${CONFIG_PATH}/train_RPP.yaml ${CHECKPOINT_SAVE_PATH}/ckpt_0/PCB-20_32.ckpt
else
    distribute_train ${CONFIG_PATH}/train_PCB.yaml $PRETRAINED_CKPT_PATH
    wait
    distribute_train ${CONFIG_PATH}/train_RPP.yaml ${CHECKPOINT_SAVE_PATH}/ckpt_0/PCB-40_14.ckpt
    wait
    distribute_train ${CONFIG_PATH}/finetune_RPP.yaml ${CHECKPOINT_SAVE_PATH}/ckpt_0/RPP-45_14.ckpt
fi
